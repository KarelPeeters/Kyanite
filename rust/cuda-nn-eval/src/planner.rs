use std::collections::{HashMap, HashSet};

use bytemuck::cast_slice;
use itertools::Itertools;

use cuda_sys::bindings::{cudnnActivationMode_t, cudnnOpTensorOp_t};
use cuda_sys::wrapper::descriptor::{ActivationDescriptor, ConvolutionDescriptor, TensorOpDescriptor};
use cuda_sys::wrapper::group::{BatchedMatMulArgs, FusedConvolutionArgs, TensorOpArgs};
use cuda_sys::wrapper::operation::STANDARD_CONV_ALGO;
use nn_graph::graph::{ElementOp, Graph, Operation, SliceRange, Value};
use nn_graph::optimizer::find_single_use_values;
use nn_graph::shape::{ConcreteShape, Size};

use crate::executor::{Handles, Step};
use crate::tensor::DeviceTensor;

pub(crate) struct Planner<'a> {
    handles: &'a Handles,
    graph: &'a Graph,
    batch_size: usize,

    // all values that are only used once in the graph (and are thus candidates for fusing)
    single_use: HashSet<Value>,

    map: HashMap<Value, DeviceTensor>,
    plan: Vec<Step>,
}

impl<'a> Planner<'a> {
    pub(crate) fn new(handles: &'a Handles, graph: &'a Graph, batch_size: usize) -> Self {
        let single_use = find_single_use_values(graph);

        Planner {
            handles,
            graph,
            batch_size,
            single_use,
            map: Default::default(),
            plan: vec![],
        }
    }

    pub fn finish(self) -> Vec<Step> {
        self.plan
    }

    pub fn visit(&mut self, value: Value) -> DeviceTensor {
        if let Some(result) = self.map.get(&value) {
            return result.clone();
        }

        if self.graph[value].shape.rank() == 4 {
            if let Some(result) = self.visit_fused_conv(value) {
                let prev = self.map.insert(value, result.clone());
                assert!(prev.is_none());
                return result;
            }
        }

        let result_info = &self.graph[value];
        let result_shape = result_info.shape.eval(self.batch_size);

        let result: DeviceTensor = match &result_info.operation {
            &Operation::Input { index: _ } => {
                // just allocate space, the user is responsible for writing the data
                self.alloc_tensor(result_shape)
            }
            Operation::Constant { data } => {
                let result = self.alloc_tensor(result_shape);
                unsafe {
                    result.copy_simple_from_host(cast_slice(&**data));
                }
                result
            }
            &Operation::View { input } => {
                let input_tensor = self.visit(input);

                let new_shape = input_tensor
                    .shape
                    .view(result_shape.dims.clone())
                    .unwrap_or_else(|| panic!("Cannot view shape {:?} as {:?}", input_tensor.shape, result_shape));

                DeviceTensor::new(input_tensor.ptr.clone(), new_shape)
            }
            &Operation::Permute { input, ref permutation } => self.visit(input).permute(permutation),
            &Operation::Slice { input, axis, range } => self.visit(input).slice(axis, range),
            &Operation::Flip { input, axis } => self.visit(input).flip(axis),
            &Operation::Gather { input, axis, indices } => {
                let input = self.visit(input);
                let indices = self.visit(indices);

                let output = self.alloc_tensor(result_shape);

                self.plan.push(Step::Gather {
                    input,
                    axis,
                    indices,
                    output: output.clone(),
                });
                output
            }
            &Operation::Concat { ref inputs, axis } => {
                let result = self.alloc_tensor(result_shape);
                let inputs = inputs.iter().map(|&x| self.visit(x)).collect_vec();

                // copy each input into the corresponding slice of the output
                let mut curr_start = 0;

                for input in inputs {
                    let curr_size = input.shape.shape()[axis];
                    let curr_range = SliceRange::new(curr_start, curr_start + curr_size, 1);

                    let curr_result = result.slice(axis, curr_range);
                    let args = curr_result.copy_from_as_tensor_op(&input);
                    self.plan.push(Step::TensorOp { args });

                    curr_start += curr_size;
                }

                result
            }
            &Operation::Conv { .. } => {
                unreachable!("conv should have been handled earlier by the fuser")
            }
            &Operation::MatMul { left, right } => {
                let left = self.visit(left);
                let right = self.visit(right);

                assert!(left.shape.rank() == 3 && right.shape.rank() == 3);
                let batch_size = left.shape.shape()[0];
                let m = left.shape.shape()[1];
                let k = left.shape.shape()[2];
                let n = right.shape.shape()[2];

                // construct a result tensor with col-major strides
                let result_transposed = self.alloc_tensor(ConcreteShape::new(vec![batch_size, n, m]));
                let result = result_transposed.permute(&[0, 2, 1]);

                let args = BatchedMatMulArgs {
                    m: m as i32,
                    n: n as i32,
                    k: k as i32,
                    alpha: 1.0,
                    beta: 0.0,
                    a: left.to_mat_mul_arg(),
                    b: right.to_mat_mul_arg(),
                    c: result.to_mat_mul_arg(),
                    batch_count: batch_size as i32,
                };

                self.plan.push(Step::MatMul { args });

                result
            }
            &Operation::Element { left, right, op } => {
                let (op, negate_right) = match op {
                    ElementOp::Add => (cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD, false),
                    ElementOp::Sub => (cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD, true),
                    ElementOp::Mul => (cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MUL, false),
                    ElementOp::Div => todo!("GPU elementwise division not yet supported"),
                    ElementOp::Min => (cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MIN, false),
                    ElementOp::Max => (cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MAX, false),
                };

                self.visit_op(result_shape, left, right, op, negate_right)
            }
        };

        assert_eq!(
            result.shape.shape(),
            result_info.shape.eval(self.batch_size).dims,
            "Got wrong result shape"
        );
        let prev = self.map.insert(value, result.clone());
        assert!(prev.is_none());

        result
    }

    fn visit_fused_conv(&mut self, value: Value) -> Option<DeviceTensor> {
        let mut curr = value;
        let graph = self.graph;

        // relu(curr)?
        let act_mode = if let &Operation::Element {
            left,
            right,
            op: ElementOp::Max,
        } = &graph[curr].operation
        {
            if !self.single_use.contains(&left) || !graph.is_const_filled_with(right, 0.0) {
                return None;
            }
            curr = left;
            cudnnActivationMode_t::CUDNN_ACTIVATION_RELU
        } else {
            cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY
        };

        let mut bias = None;
        let mut res = None;

        while let &Operation::Element {
            left,
            right,
            op: ElementOp::Add,
        } = &graph[curr].operation
        {
            if !self.single_use.contains(&left) {
                return None;
            }

            //TODO check that left != conv input
            //TODO check in advance whether outputs will be densely strided, instead of asserting it at the end
            //TODO try visiting both left and right for the continuation

            if graph[left].shape == graph[right].shape {
                // has to be res
                if res.is_none() {
                    res = Some(right);
                } else {
                    return None;
                }
            } else if graph[left].shape.keep(1, Size::ONE) == graph[right].shape {
                // has to be bias
                if bias.is_none() {
                    bias = Some(right);
                } else {
                    return None;
                }
            } else {
                return None;
            }

            curr = left;
        }

        if let &Operation::Conv { input, filter, details } = &graph[curr].operation {
            if let Some(res) = res {
                if graph[input].shape.eval(self.batch_size) != graph[res].shape.eval(self.batch_size) {
                    return None;
                }
            }

            let input = self.visit(input);
            let filter = self.visit(filter);
            let bias = bias.map(|bias| self.visit(bias));
            let res = res.map(|res| self.visit(res));

            if let Some(res) = &res {
                //TODO this should be checked before we actually start fusing
                assert_eq!(
                    res.shape, input.shape,
                    "Input and res shapes and strides (!) must match.",
                );
            }

            let bias = bias.unwrap_or_else(|| {
                let bias_shape = ConcreteShape::new(vec![1, details.output_channels, 1, 1]);
                self.alloc_zero_tensor(bias_shape)
            });

            let output_shape = graph[curr].shape.eval(self.batch_size);
            let output = self.alloc_tensor(output_shape);

            let input_desc = input.shape.descriptor();
            let output_desc = output.shape.descriptor();
            let filter_desc = filter.shape.filter_descriptor();

            let conv_desc = ConvolutionDescriptor::new(details.padding_y as i32, details.padding_x as i32, 1, 1, 1, 1);

            let algo = STANDARD_CONV_ALGO;
            let work_size =
                conv_desc.workspace_size(&self.handles.cudnn, algo, &input_desc, &filter_desc, &output_desc);
            let work_ptr = self.handles.cudnn.device().alloc(work_size);

            let act_desc = ActivationDescriptor::new(act_mode, 0.0);

            let args = FusedConvolutionArgs {
                conv_desc,
                algo,
                work_ptr,
                work_size_bytes: work_size,
                filter_desc,
                filter_ptr: filter.ptr,
                input_desc,
                input_ptr: input.ptr,
                res_ptr: res.map(|res| res.ptr),
                bias_desc: bias.shape.descriptor(),
                bias_ptr: bias.ptr,
                act_desc,
                output_desc,
                output_ptr: output.ptr.clone(),
            };

            self.plan.push(Step::Conv { args });
            Some(output)
        } else {
            None
        }
    }

    fn visit_op(
        &mut self,
        result_shape: ConcreteShape,
        left: Value,
        right: Value,
        op: cudnnOpTensorOp_t,
        negate_right: bool,
    ) -> DeviceTensor {
        let op_desc = TensorOpDescriptor::new(op);
        let alpha_2 = if negate_right { -1.0 } else { 1.0 };

        let output = self.alloc_tensor(result_shape);

        let left = self.visit(left);
        let right = self.visit(right);

        let args = TensorOpArgs {
            op_desc,
            alpha_1: 1.0,
            input_1_desc: left.shape.descriptor(),
            input_1_ptr: left.ptr.clone(),
            alpha_2,
            input_2_desc: right.shape.descriptor(),
            input_2_ptr: right.ptr.clone(),
            beta: 0.0,
            output_desc: output.shape.descriptor(),
            output_ptr: output.ptr.clone(),
        };

        self.plan.push(Step::TensorOp { args });
        output
    }

    fn alloc_tensor(&mut self, shape: ConcreteShape) -> DeviceTensor {
        DeviceTensor::alloc_simple(self.handles.cudnn.device(), shape.dims)
    }

    fn alloc_zero_tensor(&mut self, shape: ConcreteShape) -> DeviceTensor {
        let result = self.alloc_tensor(shape);
        unsafe {
            result.copy_simple_from_host(&vec![0.0; result.shape.size()]);
        }
        result
    }
}
