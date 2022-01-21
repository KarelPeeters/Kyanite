use std::collections::{HashMap, HashSet};

use bytemuck::cast_slice;
use itertools::Itertools;

use cuda_sys::bindings::{cublasOperation_t, cudnnActivationMode_t, cudnnOpTensorOp_t};
use cuda_sys::wrapper::descriptor::{ActivationDescriptor, ConvolutionDescriptor, TensorOpDescriptor};
use cuda_sys::wrapper::group::{BatchedMatMulArgs, FusedConvolutionArgs, MatMulArg, TensorOpArgs};
use cuda_sys::wrapper::mem::device::DeviceMem;
use cuda_sys::wrapper::operation::STANDARD_CONV_ALGO;
use nn_graph::graph::{ElementOp, Graph, Operation, Value};
use nn_graph::optimizer::find_single_use_values;
use nn_graph::shape::ConcreteShape;

use crate::executor::{Handles, Step};
use crate::shape::StridedShape;
use crate::tensor::Tensor;

pub(crate) struct Planner<'a> {
    handles: &'a Handles,
    graph: &'a Graph,
    batch_size: usize,

    // all values that are only used once in the graph (and are thus candidates for fusing)
    single_use: HashSet<Value>,

    map: HashMap<Value, Tensor>,
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

    pub fn copy_output(&mut self, index: usize, value: Value) -> Tensor {
        let tensor = self.map.get(&value).unwrap();
        self.plan.push(Step::CopyOutput { index, tensor: tensor.view() });
        tensor.view()
    }

    pub fn finish(self) -> Vec<Step> {
        self.plan
    }

    pub fn visit(&mut self, value: Value) -> Tensor {
        if let Some(result) = self.map.get(&value) {
            return result.view();
        }

        if let Some(result) = self.visit_fused_conv(value) {
            let prev = self.map.insert(value, result.view());
            assert!(prev.is_none());
            return result;
        }

        let result_info = &self.graph[value];
        let result_shape = result_info.shape.eval(self.batch_size);

        let result: Tensor = match &result_info.operation {
            &Operation::Input { index } => {
                let result = self.alloc_tensor(result_shape);
                self.plan.push(Step::CopyInput { index, mem: result.mem.view() });
                result
            }
            Operation::Constant { data } => {
                let result = self.alloc_tensor(result_shape);
                unsafe {
                    result.mem.copy_from_host(cast_slice(&**data));
                }
                result
            }
            &Operation::View { input } => {
                let input_tensor = self.visit(input);

                let new_shape = input_tensor.shape.view(result_shape.dims.clone())
                    .unwrap_or_else(|| panic!("Cannot view shape {:?} as {:?}", input_tensor.shape, result_shape));

                Tensor::new(input_tensor.mem.view(), new_shape)
            }
            &Operation::Permute { input, ref permutation } => {
                let input_tensor = self.visit(input);
                input_tensor.permute(permutation)
            }
            &Operation::Slice { input, axis, start, end } => {
                let input_tensor = self.visit(input);
                input_tensor.slice(axis, start, end)
            }
            &Operation::Gather { input, axis, indices } => {
                let input = self.visit(input);
                let indices = self.visit(indices);

                let output = self.alloc_tensor(result_shape);

                self.plan.push(Step::Gather { input, axis, indices, output: output.view() });
                output
            }
            &Operation::Concat { ref inputs, axis } => {
                let result = self.alloc_tensor(result_shape);
                let inputs = inputs.iter().map(|&x| self.visit(x)).collect_vec();

                // copy each input into the right slice of the output
                let mut curr_start = 0;
                for input in inputs {
                    let curr_size = input.shape.shape()[axis];
                    let result_slice = result.slice(axis, curr_start, curr_start + curr_size);
                    let zero = self.alloc_zero_tensor(ConcreteShape::new(vec![1; result_slice.shape.rank()]));

                    let args = TensorOpArgs {
                        op_desc: TensorOpDescriptor::new(cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD),
                        alpha_1: 1.0,
                        input_1_desc: input.descriptor(),
                        input_1_mem: input.mem,
                        alpha_2: 0.0,
                        input_2_desc: zero.descriptor(),
                        input_2_mem: zero.mem,
                        beta: 0.0,
                        output_desc: result_slice.descriptor(),
                        output_mem: result_slice.mem.view(),
                    };
                    self.plan.push(Step::TensorOp { args });

                    curr_start += curr_size;
                }

                result
            }
            &Operation::Conv { .. } =>
                unreachable!("conv should have been handled earlier by the fuser"),
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
                    a: to_mat_mul_arg(&left),
                    b: to_mat_mul_arg(&right),
                    c: to_mat_mul_arg(&result),
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

        assert_eq!(result.shape.shape(), result_info.shape.eval(self.batch_size).dims, "Got wrong result shape");
        let prev = self.map.insert(value, result.view());
        assert!(prev.is_none());

        result
    }

    fn visit_fused_conv(&mut self, value: Value) -> Option<Tensor> {
        let mut curr = value;
        let graph = self.graph;

        // relu(curr)?
        let act_mode = if let &Operation::Element { left, right, op: ElementOp::Max } = &graph[curr].operation {
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

        while let &Operation::Element { left, right, op: ElementOp::Add } = &graph[curr].operation {
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
            } else if graph[left].shape.all_ones_except(1) == graph[right].shape {
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
                assert_eq!(res.shape, input.shape, "Input and res shapes and strides (!) must match.", );
            }

            let bias = bias.unwrap_or_else(|| {
                let bias_shape = ConcreteShape::new(vec![1, details.output_channels, 1, 1]);
                self.alloc_zero_tensor(bias_shape)
            });

            let output_shape = graph[curr].shape.eval(self.batch_size);
            let output = self.alloc_tensor(output_shape);

            let input_desc = input.descriptor();
            let output_desc = output.descriptor();
            let filter_desc = filter.filter_descriptor();

            let conv_desc = ConvolutionDescriptor::new(
                details.padding_y as i32, details.padding_x as i32,
                1, 1, 1, 1,
            );

            let algo = STANDARD_CONV_ALGO;
            let work_size_bytes = conv_desc.workspace_size(&self.handles.cudnn, algo, &input_desc, &filter_desc, &output_desc);
            let work_mem = DeviceMem::alloc(work_size_bytes, self.handles.cudnn.device());

            let act_desc = ActivationDescriptor::new(act_mode, 0.0);

            let args = FusedConvolutionArgs {
                conv_desc,
                algo,
                work_mem,
                filter_desc,
                filter_mem: filter.mem,
                input_desc,
                input_mem: input.mem,
                res_mem: res.map(|res| res.mem),
                bias_desc: bias.descriptor(),
                bias_mem: bias.mem,
                act_desc,
                output_desc,
                output_mem: output.mem.view(),
            };

            self.plan.push(Step::Conv { args });
            Some(output)
        } else {
            None
        }
    }

    fn visit_op(&mut self, result_shape: ConcreteShape, left: Value, right: Value, op: cudnnOpTensorOp_t, negate_right: bool) -> Tensor {
        let op_desc = TensorOpDescriptor::new(op);
        let alpha_2 = if negate_right { -1.0 } else { 1.0 };

        let output = self.alloc_tensor(result_shape);

        let left = self.visit(left);
        let right = self.visit(right);

        let args = TensorOpArgs {
            op_desc,
            alpha_1: 1.0,
            input_1_desc: left.descriptor(),
            input_1_mem: left.mem.view(),
            alpha_2,
            input_2_desc: right.descriptor(),
            input_2_mem: right.mem.view(),
            beta: 0.0,
            output_desc: output.descriptor(),
            output_mem: output.mem.view(),
        };

        self.plan.push(Step::TensorOp { args });
        output
    }

    fn alloc_tensor(&mut self, shape: ConcreteShape) -> Tensor {
        let shape = StridedShape::new_simple(shape.dims);
        Tensor::new(
            DeviceMem::alloc(shape.strided_size() * 4, self.handles.cudnn.device()),
            shape,
        )
    }

    fn alloc_zero_tensor(&mut self, shape: ConcreteShape) -> Tensor {
        let size = shape.size();
        let result = self.alloc_tensor(shape);
        unsafe {
            result.mem.copy_from_host(cast_slice(&vec![0; size]));
        }
        result
    }
}

fn to_mat_mul_arg(tensor: &Tensor) -> MatMulArg {
    assert_eq!(tensor.shape.rank(), 3);

    let inner_shape = StridedShape::new(
        tensor.shape.shape()[1..].to_vec(),
        tensor.shape.strides()[1..].to_vec(),
    );

    // whether the strides are col-major (true) or row-major (false)
    let col_major = if inner_shape.has_simple_strides() {
        false
    } else if inner_shape.permute(&[1, 0]).has_simple_strides() {
        true
    } else {
        panic!("For now GPU matmul operand must be either col- or row-major, got {:?}", tensor)
    };

    let lead_axis = if col_major { 1 } else { 2 };

    MatMulArg {
        mem: tensor.mem.view(),
        trans: if col_major { cublasOperation_t::CUBLAS_OP_N } else { cublasOperation_t::CUBLAS_OP_T },
        ld: tensor.shape.shape()[lead_axis] as i32,
        stride: tensor.shape.strides()[0] as i64,
    }
}
