use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};

use bytemuck::cast_slice;
use internal_iterator::InternalIterator;
use itertools::Itertools;

use cuda_sys::bindings::cudnnActivationMode_t;
use cuda_sys::wrapper::descriptor::{ActivationDescriptor, ConvolutionDescriptor};
use cuda_sys::wrapper::group::{BatchedMatMulArgs, FusedConvolutionArgs};
use cuda_sys::wrapper::handle::Device;
use cuda_sys::wrapper::mem::device::DevicePtr;
use cuda_sys::wrapper::operation::STANDARD_CONV_ALGO;
use nn_graph::graph::{BinaryOp, Graph, Operation, SliceRange, UnaryOp, Value};
use nn_graph::optimizer::core::find_hidden_values_used_once;
use nn_graph::shape::{ConcreteShape, Size};

use crate::autokernel::reduce::{ReduceCode, ReduceKernel};
use crate::autokernel::scalar::ScalarKernel;
use crate::device_tensor::DeviceTensor;
use crate::executor::Handles;
use crate::offset_tensor::{OffsetPtr, PtrTensor};
use crate::shape::StridedShape;
use crate::step::{GatherArgs, ReduceOpArgs, ScalarOpArgs, Step};

#[derive(Debug, Clone)]
enum PlanBuffer {
    Dedicated(DevicePtr),
    Shared { index: usize },
    Zero,
}

#[derive(Debug, Clone)]
struct PlanPtr {
    buffer: PlanBuffer,
    offset_bytes: isize,
}

type PlanTensor = PtrTensor<PlanPtr>;
type PlanStep = Step<PlanPtr>;
type ExecStep = Step<DevicePtr>;

pub(crate) struct Planner<'a> {
    handles: &'a Handles,
    graph: &'a Graph,
    batch_size: usize,

    // all values that are only used once in the graph (and are thus candidates for fusing)
    fuse_candidates: HashSet<Value>,

    shared_buffers_size_in_bytes: Vec<usize>,
    max_zero_size: usize,

    map: HashMap<Value, PlanTensor>,
    plan: Vec<PlanStep>,

    dedicated_bytes: usize,
}

#[derive(Debug)]
pub struct Plan {
    pub inputs: Vec<DeviceTensor>,
    pub outputs: Vec<DeviceTensor>,
    pub steps: Vec<Step<DevicePtr>>,
    pub mem_usage: MemoryUsage,
}

#[derive(Debug)]
pub struct MemoryUsage {
    pub dedicated_bytes: usize,
    pub shared_bytes: usize,
    pub zero_bytes: usize,
}

impl<'a> Planner<'a> {
    pub fn plan(handles: &'a Handles, graph: &'a Graph, batch_size: usize) -> Plan {
        let mut planner = Planner::new(&handles, graph, batch_size);

        // allocate inputs (even if they're not actually used)
        let inputs = graph.inputs().iter().map(|&input| planner.visit(input)).collect_vec();

        // collect outputs, this recursively plans all the necessary operations
        let outputs = graph
            .outputs()
            .iter()
            .map(|&output| planner.visit_ensure_simple_strides(output))
            .collect_vec();

        let buffer_count = planner.shared_buffers_size_in_bytes.len();
        let step_count = planner.plan.len();
        let mut shared_bytes = 0;

        // determine live ranges for shared tensors
        let live_ranges = {
            let mut live_ranges = vec![(step_count, 0); buffer_count];

            for (si, step) in planner.plan.iter().enumerate() {
                step.ptr_operands().for_each(|op| {
                    if let &PlanBuffer::Shared { index } = &op.buffer {
                        let (lower, upper) = &mut live_ranges[index];
                        *lower = min(*lower, si);
                        *upper = max(*upper, si);
                    }
                })
            }

            // consider outputs live at the end
            for output in &outputs {
                if let &PlanBuffer::Shared { index } = &output.ptr().buffer {
                    let (_, upper) = &mut live_ranges[index];
                    *upper = step_count;
                }
            }

            // consider inputs live at the start
            for input in &inputs {
                if let &PlanBuffer::Shared { index } = &input.ptr().buffer {
                    let (lower, _) = &mut live_ranges[index];
                    *lower = 0;
                }
            }

            live_ranges
        };

        // actually allocate shared tensors
        let mut free_allocations: HashMap<usize, Vec<DevicePtr>> = Default::default();
        let device = planner.device();

        let mut shared_allocations: Vec<Option<DevicePtr>> = vec![None; buffer_count];

        for si in 0..step_count {
            for (ti, &(start, _)) in live_ranges.iter().enumerate() {
                if start == si {
                    // allocate the given tensor
                    let size_bytes = planner.shared_buffers_size_in_bytes[ti];
                    let vec = free_allocations.entry(size_bytes).or_insert_with(Vec::new);
                    let ptr = vec.pop().unwrap_or_else(|| {
                        shared_bytes += size_bytes;
                        device.alloc(size_bytes)
                    });

                    assert!(shared_allocations[ti].is_none());
                    shared_allocations[ti] = Some(ptr);
                }
            }

            for (ti, &(_, end)) in live_ranges.iter().enumerate() {
                if end == si {
                    // free the given tensor
                    let size_bytes = planner.shared_buffers_size_in_bytes[ti];
                    let vec = free_allocations.get_mut(&size_bytes).unwrap();

                    let ptr = shared_allocations[ti].clone().unwrap();
                    vec.push(ptr);
                }
            }
        }

        // collect shared allocations
        let shared_allocations = shared_allocations
            .into_iter()
            .enumerate()
            .map(|(i, ptr)| {
                ptr.unwrap_or_else(|| {
                    // allocate leftover buffers that are never used before the end (eg. empty concat output tensors)
                    let size_bytes = planner.shared_buffers_size_in_bytes[i];
                    shared_bytes += size_bytes;
                    device.alloc(size_bytes)
                })
            })
            .collect_vec();

        // allocate (single) zero tensor
        let zero_bytes = 4 * planner.max_zero_size;
        let zero_allocation = device.alloc(zero_bytes);
        unsafe {
            zero_allocation.copy_linear_from_host(cast_slice(&vec![0f32; planner.max_zero_size]));
        }

        let mem_usage = MemoryUsage {
            dedicated_bytes: planner.dedicated_bytes,
            shared_bytes,
            zero_bytes,
        };

        // realize planned tensors and steps
        let ctx = RealizationContext {
            shared_allocations,
            zero_allocation,
        };

        Plan {
            inputs: inputs.into_iter().map(|t| ctx.realize_tensor(t)).collect(),
            outputs: outputs.into_iter().map(|t| ctx.realize_tensor(t)).collect(),
            steps: planner
                .plan
                .into_iter()
                .map(|step| ctx.realize_step(step))
                .collect_vec(),
            mem_usage,
        }
    }

    fn new(handles: &'a Handles, graph: &'a Graph, batch_size: usize) -> Self {
        let fuse_candidates = find_hidden_values_used_once(graph).collect();

        Planner {
            handles,
            graph,
            batch_size,
            fuse_candidates,
            shared_buffers_size_in_bytes: vec![],
            map: Default::default(),
            plan: vec![],
            max_zero_size: 0,
            dedicated_bytes: 0,
        }
    }

    fn visit_ensure_simple_strides(&mut self, value: Value) -> PlanTensor {
        let result = self.visit(value);

        if result.shape().has_simple_strides() {
            result
        } else {
            let shape = ConcreteShape::new(result.shape().shape().to_vec());
            let new = self.alloc_tensor_shared(shape);
            self.plan_copy_tensor(&result, &new);
            new
        }
    }

    fn visit(&mut self, value: Value) -> PlanTensor {
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

        let result: PlanTensor = match &result_info.operation {
            &Operation::Input { index: _ } => self.alloc_tensor_shared(result_shape),
            Operation::Constant { data } => {
                let result = self.alloc_tensor_dedicated(result_shape);
                unsafe {
                    result.copy_simple_from_host(cast_slice(&**data));
                }
                result.map_ptr(PlanPtr::from)
            }
            &Operation::View { input } => {
                let input_tensor = self.visit(input);

                // try simple view operation, otherwise restride to dense and then copy
                // TODO if we want the output to be simple strided immediately we need separate input/output shapes
                //    in the scalar kernel
                input_tensor.view(result_shape.dims.clone()).unwrap_or_else(|_| {
                    let input_shape = ConcreteShape::new(input_tensor.shape().shape().to_vec());
                    let result = self.alloc_tensor_shared(input_shape);
                    self.plan_copy_tensor(&input_tensor, &result);
                    result.view(result_shape.dims.clone()).unwrap()
                })
            }
            &Operation::Broadcast { input } => {
                let input_tensor = self.visit(input);
                input_tensor.broadcast(result_shape.dims.clone())
            }
            &Operation::Permute { input, ref permutation } => self.visit(input).permute(permutation),
            &Operation::Slice { input, axis, range } => self.visit(input).slice(axis, range),
            &Operation::Flip { input, axis } => self.visit(input).flip(axis),
            &Operation::Gather { input, axis, indices } => {
                let input = self.visit(input);
                let indices = self.visit(indices);
                let output = self.alloc_tensor_shared(result_shape);

                let args = GatherArgs {
                    input,
                    axis,
                    indices,
                    output: output.clone(),
                };

                self.plan.push(PlanStep::Gather(args));

                output
            }
            &Operation::Concat { ref inputs, axis } => {
                let result = self.alloc_tensor_shared(result_shape);
                let inputs = inputs.iter().map(|&x| self.visit(x)).collect_vec();

                // copy each input into the corresponding slice of the output
                let mut curr_start = 0;

                for input in inputs {
                    let curr_size = input.shape().shape()[axis];
                    let curr_range = SliceRange::simple(curr_start, curr_start + curr_size);

                    let curr_result = result.slice(axis, curr_range);
                    self.plan_copy_tensor(&input, &curr_result);

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

                assert!(left.shape().rank() == 3 && right.shape().rank() == 3);
                let batch_size = left.shape().shape()[0];
                let m = left.shape().shape()[1];
                let k = left.shape().shape()[2];
                let n = right.shape().shape()[2];

                // construct a result tensor with col-major strides
                let result_transposed = self.alloc_tensor_shared(ConcreteShape::new(vec![batch_size, n, m]));
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

                self.plan.push(PlanStep::MatMul(args));

                result
            }
            &Operation::Unary { input, op } => {
                let operation = match op {
                    UnaryOp::Sqrt => "*x0 = sqrt(*x1);",
                    UnaryOp::Exp => "*x0 = exp(*x1);",
                };

                let input = self.visit(input);
                let output = self.alloc_tensor_shared(result_shape);

                self.plan_scalar_op(operation, vec![output.clone(), input]);

                output
            }
            &Operation::Binary { left, right, op } => {
                let op_str = format!("*x0 = {};", binary_op_str(op, "*x1", "*x2"));

                let left = self.visit(left);
                let right = self.visit(right);
                let output = self.alloc_tensor_shared(result_shape);

                self.plan_scalar_op(&op_str, vec![output.clone(), left, right]);

                output
            }
            &Operation::Softmax { .. } => todo!("GPU softmax"),
            &Operation::Reduce { input, ref axes, op } => {
                let result_size = result_shape.size();

                let input = self.visit(input);
                let output = self.alloc_tensor_shared(result_shape);

                let identity = op.identity();
                let (operation, is_mean) = op.operation();
                let scale = if is_mean {
                    result_size as f32 / input.shape().size() as f32
                } else {
                    1.0
                };

                let code = ReduceCode {
                    ty: "float".to_owned(),
                    identity: format!("{}", identity).replace("inf", "(1.0/0.0)"),
                    operation: binary_op_str(operation, "curr", "x"),
                    post_process: format!("curr * {}", scale),
                };

                let capability = self.device().compute_capability();
                let kernel = ReduceKernel::new(capability, code, input.shape(), output.shape(), axes);

                let args = ReduceOpArgs {
                    kernel,
                    input,
                    output: output.clone(),
                };
                self.plan.push(PlanStep::ReduceOp(args));

                output
            }
        };

        self.insert_mapping(value, result.clone());
        result
    }

    fn insert_mapping(&mut self, value: Value, result: PlanTensor) {
        assert_eq!(
            result.shape().shape(),
            self.graph[value].shape.eval(self.batch_size).dims,
            "Got wrong result shape"
        );
        let prev = self.map.insert(value, result);
        assert!(prev.is_none());
    }

    fn visit_fused_conv(&mut self, value: Value) -> Option<PlanTensor> {
        let mut curr = value;
        let graph = self.graph;

        // relu(curr)?
        let act_mode = if let &Operation::Binary {
            left,
            right,
            op: BinaryOp::Max,
        } = &graph[curr].operation
        {
            if !self.fuse_candidates.contains(&left) || !graph.is_const_filled_with(right, 0.0) {
                return None;
            }
            curr = left;
            cudnnActivationMode_t::CUDNN_ACTIVATION_RELU
        } else {
            cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY
        };

        let mut bias = None;
        let mut res = None;

        while let &Operation::Binary {
            left,
            right,
            op: BinaryOp::Add,
        } = &graph[curr].operation
        {
            if !self.fuse_candidates.contains(&left) {
                return None;
            }

            //TODO check that left != conv input
            //TODO try visiting both left and right for the continuation

            let is_res = graph[left].shape == graph[right].shape;

            let mut bias_inner = None;

            if let Operation::Broadcast { input: right_inner } = graph[right].operation {
                if graph[left].shape.keep(1, Size::ONE) == graph[right_inner].shape {
                    bias_inner = Some(right_inner);
                }
            }

            if let Some(bias_inner) = bias_inner {
                // has to be bias
                if bias.is_none() {
                    bias = Some(bias_inner);
                } else {
                    return None;
                }
            } else if is_res {
                // has to be res
                if res.is_none() {
                    res = Some(right);
                } else {
                    return None;
                }
            } else {
                return None;
            }

            curr = left;
        }

        if let &Operation::Conv { input, filter, details } = &graph[curr].operation {
            // collect input tensors
            if let Some(res) = res {
                if graph[input].shape != graph[res].shape {
                    // rejecting the conv here is fine,
                    //   since it will just be retried later without the following res addition
                    return None;
                }
            }

            // TODO for conv the input & res need simple strides, what about filter and bias?
            let input = self.visit_ensure_simple_strides(input);
            let filter = self.visit(filter);
            let bias = bias.map(|bias| self.visit(bias));
            let res = res.map(|res| self.visit_ensure_simple_strides(res));

            let bias = bias.unwrap_or_else(|| {
                let bias_shape = ConcreteShape::new(vec![1, details.output_channels, 1, 1]);
                self.alloc_tensor_zero(bias_shape)
            });

            let output_shape = graph[curr].shape.eval(self.batch_size);
            let output = self.alloc_tensor_shared(output_shape);

            // build descriptors
            let input_desc = input.shape().descriptor();
            let bias_desc = bias.shape().descriptor();
            let filter_desc = filter.shape().filter_descriptor();
            let output_desc = output.shape().descriptor();

            let conv_desc = ConvolutionDescriptor::new(details.padding_y as i32, details.padding_x as i32, 1, 1, 1, 1);
            let act_desc = ActivationDescriptor::new(act_mode, 0.0);
            let algo = STANDARD_CONV_ALGO;

            let work_size_bytes =
                conv_desc.workspace_size(&self.handles.cudnn, algo, &input_desc, &filter_desc, &output_desc);
            let work_ptr = self.alloc_buffer_shared(work_size_bytes);

            // put everything together
            let args = FusedConvolutionArgs {
                conv_desc,
                algo,
                work_ptr,
                work_size_bytes,
                filter_desc,
                filter_ptr: filter.into_ptr(),
                input_desc,
                input_ptr: input.into_ptr(),
                res_ptr: res.map(|res| res.into_ptr()),
                bias_desc,
                bias_ptr: bias.into_ptr(),
                act_desc,
                output_desc,
                output_ptr: output.ptr().clone(),
            };
            self.plan.push(PlanStep::Conv(args));

            Some(output)
        } else {
            None
        }
    }

    fn plan_copy_tensor(&mut self, old: &PlanTensor, new: &PlanTensor) {
        assert_eq!(old.shape().shape(), new.shape().shape());

        self.plan_scalar_op("*x0 = *x1;", vec![new.clone(), old.clone()]);
    }

    fn plan_scalar_op(&mut self, operation: &str, operands: Vec<PlanTensor>) {
        let capability = self.device().compute_capability();
        let shapes = operands.iter().map(|operand| operand.shape().clone()).collect_vec();

        let kernel = ScalarKernel::new_for_shapes(capability, operation, &shapes);

        let args = ScalarOpArgs { kernel, operands };
        self.plan.push(PlanStep::ScalarOp(args));
    }

    fn alloc_tensor_dedicated(&mut self, shape: ConcreteShape) -> DeviceTensor {
        self.dedicated_bytes += 4 * shape.size();
        DeviceTensor::alloc_simple(self.device(), shape.dims)
    }

    fn alloc_buffer_shared(&mut self, size_in_bytes: usize) -> PlanPtr {
        let index = self.shared_buffers_size_in_bytes.len();
        self.shared_buffers_size_in_bytes.push(size_in_bytes);
        PlanPtr::from_parts(PlanBuffer::Shared { index }, 0)
    }

    fn alloc_tensor_shared(&mut self, shape: ConcreteShape) -> PlanTensor {
        let buffer = self.alloc_buffer_shared(4 * shape.size());
        let shape = StridedShape::new_simple(shape.dims);
        PlanTensor::from_parts(buffer, shape)
    }

    fn alloc_tensor_zero(&mut self, shape: ConcreteShape) -> PlanTensor {
        let shape = StridedShape::new_simple(shape.dims);
        let size = shape.size();
        self.max_zero_size = max(self.max_zero_size, size);
        PlanTensor::from_parts(PlanPtr::from_parts(PlanBuffer::Zero, 0), shape)
    }

    fn device(&self) -> Device {
        self.handles.device()
    }
}

#[derive(Debug)]
struct RealizationContext {
    shared_allocations: Vec<DevicePtr>,
    zero_allocation: DevicePtr,
}

impl RealizationContext {
    fn realize_ptr(&self, ptr: PlanPtr) -> DevicePtr {
        let PlanPtr { buffer, offset_bytes } = ptr;

        let base = match buffer {
            PlanBuffer::Dedicated(buffer) => buffer,
            PlanBuffer::Shared { index } => self.shared_allocations[index].clone(),
            PlanBuffer::Zero => self.zero_allocation.clone(),
        };

        base.offset_bytes(offset_bytes)
    }

    fn realize_tensor(&self, tensor: PlanTensor) -> DeviceTensor {
        tensor.map_ptr(|ptr| self.realize_ptr(ptr))
    }

    fn realize_step(&self, step: PlanStep) -> ExecStep {
        step.map_ptrs(|ptr| self.realize_ptr(ptr))
    }
}

impl PlanPtr {
    pub fn from_parts(buffer: PlanBuffer, offset_bytes: isize) -> Self {
        Self { buffer, offset_bytes }
    }
}

impl OffsetPtr for PlanPtr {
    fn offset_bytes(self, offset_bytes: isize) -> Self {
        PlanPtr::from_parts(self.buffer, self.offset_bytes + offset_bytes)
    }
}

impl From<DevicePtr> for PlanPtr {
    fn from(ptr: DevicePtr) -> Self {
        PlanPtr::from_parts(PlanBuffer::Dedicated(ptr), 0)
    }
}

fn binary_op_str(op: BinaryOp, a: &str, b: &str) -> String {
    match op {
        BinaryOp::Add => format!("{} + {}", a, b),
        BinaryOp::Sub => format!("{} - {}", a, b),
        BinaryOp::Mul => format!("{} * {}", a, b),
        BinaryOp::Div => format!("{} / {}", a, b),
        BinaryOp::Min => format!("min({}, {})", a, b),
        BinaryOp::Max => format!("max({}, {})", a, b),
        BinaryOp::Pow => format!("powf({}, {})", a, b),
    }
}
