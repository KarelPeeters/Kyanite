use std::cmp::{max, min};
use std::collections::HashMap;

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
use nn_graph::shape::{ConcreteShape, Size};

use crate::autokernel::layernorm::LayernormKernel;
use crate::autokernel::reduce::{ReduceCode, ReduceKernel};
use crate::autokernel::scalar::ScalarKernel;
use crate::autokernel::softmax::SoftmaxKernel;
use crate::device_tensor::DeviceTensor;
use crate::executor::Handles;
use crate::offset_tensor::{OffsetPtr, PtrTensor};
use crate::shape::StridedShape;
use crate::step::{GatherArgs, LayernormOpArgs, ReduceOpArgs, ScalarOpArgs, SoftmaxOpArgs, Step, StepInfo};

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
type PlanStepInfo = StepInfo<PlanPtr>;
type ExecStepInfo = StepInfo<DevicePtr>;

pub(crate) struct Planner<'a> {
    handles: &'a Handles,
    graph: &'a Graph,
    batch_size: usize,

    shared_buffers_size_in_bytes: Vec<usize>,
    max_zero_size: usize,

    map: HashMap<Value, PlanTensor>,
    steps: Vec<PlanStepInfo>,

    dedicated_bytes: usize,
}

#[derive(Debug)]
pub struct Plan {
    pub inputs: Vec<DeviceTensor>,
    pub outputs: Vec<DeviceTensor>,
    pub steps: Vec<ExecStepInfo>,
    pub mem_usage: MemoryUsage,
}

#[derive(Debug)]
pub struct MemoryUsage {
    pub dedicated_bytes: usize,
    pub shared_bytes: usize,
    pub zero_bytes: usize,
}

/// This planner is implemented fully recursively, but this means that we can hit stack-overflows for deep graphs.
/// To solve this we "break" the recursion using this type:
/// * `Ok(T)` means the mapping was completed.
/// * `Err(other_value)` means `other_value` should be visited first,
///   after which the original value should be tried again.
type VisitResult<T> = Result<T, Value>;

impl<'a> Planner<'a> {
    pub fn plan(handles: &'a Handles, graph: &'a Graph, batch_size: usize) -> Plan {
        let mut planner = Planner::new(&handles, graph, batch_size);

        // allocate inputs (even if they're not actually used)
        let inputs = graph
            .inputs()
            .iter()
            .map(|&input| planner.visit_completely(input))
            .collect_vec();

        // collect outputs, this recursively plans all the necessary operations
        let outputs = graph
            .outputs()
            .iter()
            .enumerate()
            .map(|(oi, &output)| planner.visit_completely_ensure_simple_strides(output, &format!("output_{}", oi)))
            .collect_vec();

        let buffer_count = planner.shared_buffers_size_in_bytes.len();
        let step_count = planner.steps.len();
        let mut shared_bytes = 0;

        // determine live ranges for shared tensors
        let live_ranges = {
            let mut live_ranges = vec![(step_count, 0); buffer_count];

            for (si, step_info) in planner.steps.iter().enumerate() {
                step_info.step.ptr_operands().for_each(|op| {
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
                .steps
                .into_iter()
                .map(|PlanStepInfo { step, debug_id }| ExecStepInfo {
                    step: ctx.realize_step(step),
                    debug_id,
                })
                .collect_vec(),
            mem_usage,
        }
    }

    fn new(handles: &'a Handles, graph: &'a Graph, batch_size: usize) -> Self {
        Planner {
            handles,
            graph,
            batch_size,
            shared_buffers_size_in_bytes: vec![],
            map: Default::default(),
            steps: vec![],
            max_zero_size: 0,
            dedicated_bytes: 0,
        }
    }

    fn visit_completely_ensure_simple_strides(&mut self, value: Value, id: &str) -> PlanTensor {
        self.visit_completely(value);
        self.visit_ensure_simple_strides(value, id).unwrap()
    }

    fn visit_completely(&mut self, value: Value) -> PlanTensor {
        let mut stack = vec![value];

        loop {
            let curr = *stack.last().unwrap();

            match self.visit(curr) {
                Ok(tensor) => {
                    stack.pop().unwrap();
                    if stack.is_empty() {
                        return tensor;
                    }
                }
                Err(other_value) => stack.push(other_value),
            }
        }
    }

    fn visit(&mut self, value: Value) -> VisitResult<PlanTensor> {
        if let Some(result) = self.map.get(&value) {
            return Ok(result.clone());
        }

        if let Some(result) = self.visit_fused_conv(value)? {
            self.insert_mapping(value, result.clone());
            return Ok(result);
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
                let input_tensor = self.visit(input)?;

                // try simple view operation, otherwise restride to dense and then copy
                // TODO if we want the output to be simple strided immediately we need separate input/output shapes
                //    in the scalar kernel
                input_tensor.view(result_shape.dims.clone()).unwrap_or_else(|_| {
                    let input_shape = ConcreteShape::new(input_tensor.shape().shape().to_vec());
                    let result = self.alloc_tensor_shared(input_shape);
                    self.plan_copy_tensor(&input_tensor, &result, &result_info.debug_id);
                    result.view(result_shape.dims.clone()).unwrap()
                })
            }
            &Operation::Broadcast { input } => {
                let input_tensor = self.visit(input)?;
                input_tensor.broadcast(result_shape.dims.clone())
            }
            &Operation::Permute { input, ref permutation } => self.visit_permute(input, permutation)?,
            &Operation::Slice { input, axis, range } => self.visit(input)?.slice(axis, range),
            &Operation::Flip { input, axis } => self.visit(input)?.flip(axis),
            &Operation::Gather { input, axis, indices } => {
                let input = self.visit(input)?;
                let indices = self.visit(indices)?;
                let output = self.alloc_tensor_shared(result_shape);

                let args = GatherArgs {
                    input,
                    axis,
                    indices,
                    output: output.clone(),
                };

                self.push(PlanStep::Gather(args), &result_info.debug_id);

                output
            }
            &Operation::Concat { ref inputs, axis } => {
                let result = self.alloc_tensor_shared(result_shape);
                let inputs: Vec<PlanTensor> = inputs.iter().map(|&x| self.visit(x)).try_collect()?;

                // copy each input into the corresponding slice of the output
                let mut curr_start = 0;

                for input in inputs {
                    let curr_size = input.shape().shape()[axis];
                    let curr_range = SliceRange::simple(curr_start, curr_start + curr_size);

                    let curr_result = result.slice(axis, curr_range);
                    self.plan_copy_tensor(&input, &curr_result, &result_info.debug_id);

                    curr_start += curr_size;
                }

                result
            }
            &Operation::Conv { .. } => {
                unreachable!("conv should have been handled earlier by the fuser")
            }
            &Operation::MatMul { left, right } => self.visit_matmul(value, left, right, true)?,
            &Operation::Unary { input, op } => {
                let operation = match op {
                    UnaryOp::Sqrt => "*x0 = sqrt(*x1);",
                    UnaryOp::Exp => "*x0 = exp(*x1);",
                };

                let input = self.visit(input)?;
                let output = self.alloc_tensor_shared(result_shape);

                self.plan_scalar_op(operation, vec![output.clone(), input], &result_info.debug_id);

                output
            }
            &Operation::Binary { left, right, op } => {
                let op_str = format!("*x0 = {};", binary_op_str(op, "*x1", "*x2"));

                let left = self.visit(left)?;
                let right = self.visit(right)?;
                let output = self.alloc_tensor_shared(result_shape);

                self.plan_scalar_op(&op_str, vec![output.clone(), left, right], &result_info.debug_id);

                output
            }
            &Operation::Softmax { input, axis } => {
                let input = self.visit(input)?;
                let output = self.alloc_tensor_shared(result_shape);

                let kernel = SoftmaxKernel::new(self.device(), input.shape(), output.shape(), axis);

                let args = SoftmaxOpArgs {
                    kernel,
                    input,
                    output: output.clone(),
                };

                self.push(PlanStep::SoftmaxOp(args), &result_info.debug_id);
                output
            }
            &Operation::Layernorm { input, axis, eps } => {
                let (alpha_0, input_0, alpha_1, input_1) = self.visit_scalable_added_pair(input)?;

                if let Some(input_1) = &input_1 {
                    assert_eq!(input_0.shape(), input_1.shape());
                }

                let output = self.alloc_tensor_shared(result_shape);

                let kernel = LayernormKernel::new(
                    self.device(),
                    input_0.shape(),
                    output.shape(),
                    axis,
                    eps.into_inner(),
                    alpha_0,
                    alpha_1,
                    1.0,
                );

                let args = LayernormOpArgs {
                    kernel,
                    input0: input_0,
                    input1: input_1,
                    output: output.clone(),
                };

                self.push(PlanStep::LayernormOp(args), &result_info.debug_id);
                output
            }
            &Operation::Reduce { input, ref axes, op } => {
                let result_size = result_shape.size();

                let input = self.visit(input)?;
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

                let kernel = ReduceKernel::new(self.device(), code, input.shape(), output.shape(), axes);

                let args = ReduceOpArgs {
                    kernel,
                    input,
                    output: output.clone(),
                };
                self.push(PlanStep::ReduceOp(args), &result_info.debug_id);

                output
            }
        };

        self.insert_mapping(value, result.clone());
        Ok(result)
    }

    fn visit_scalable_added_pair(&mut self, input: Value) -> VisitResult<(f32, PlanTensor, f32, Option<PlanTensor>)> {
        let result = if let &Operation::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } = &self.graph[input].operation
        {
            let (alpha_0, input_0) = self.visit_scalable_value(left)?;
            let (alpha_1, input_1) = self.visit_scalable_value(right)?;

            if input_0.shape() != input_1.shape() {
                // fallback to scalar operation
                let total = self.visit(input)?;
                (1.0, total, 0.0, None)
            } else {
                (alpha_0, input_0, alpha_1, Some(input_1))
            }
        } else {
            let (alpha_0, input_0) = self.visit_scalable_value(input)?;
            (alpha_0, input_0, 0.0, None)
        };

        Ok(result)
    }

    fn visit_permute(&mut self, input: Value, permutation: &[usize]) -> VisitResult<PlanTensor> {
        if self.can_fuse(input) {
            // avoid creating non-densely strided output for typical attention matmul
            if permutation == &[1, 0, 2] {
                if let &Operation::MatMul { left, right } = &self.graph[input].operation {
                    let mat_mul_result = self.visit_matmul(input, left, right, false)?;
                    self.insert_mapping(input, mat_mul_result.clone());
                    return Ok(mat_mul_result.permute(permutation));
                }
            }
        }

        Ok(self.visit(input)?.permute(permutation))
    }

    fn visit_ensure_simple_strides(&mut self, value: Value, id: &str) -> VisitResult<PlanTensor> {
        let inner = self.visit(value)?;

        let result = if inner.shape().has_simple_strides() {
            inner
        } else {
            let shape = ConcreteShape::new(inner.shape().shape().to_vec());
            let new = self.alloc_tensor_shared(shape);
            self.plan_copy_tensor(&inner, &new, id);
            new
        };

        Ok(result)
    }

    fn visit_scalable_value(&mut self, value: Value) -> VisitResult<(f32, PlanTensor)> {
        if let &Operation::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } = &self.graph[value].operation
        {
            if let Some(alpha) = self.graph.as_single_const(left) {
                return Ok((alpha, self.visit(right)?));
            }
            if let Some(alpha) = self.graph.as_single_const(right) {
                return Ok((alpha, self.visit(left)?));
            }
        }

        Ok((1.0, self.visit(value)?))
    }

    fn visit_matmul(&mut self, value: Value, left: Value, right: Value, batch_first: bool) -> VisitResult<PlanTensor> {
        let result_info = &self.graph[value];
        let left = self.visit(left)?;
        let right = self.visit(right)?;

        assert!(left.shape().rank() == 3 && right.shape().rank() == 3);
        let batch_size = left.shape().shape()[0];
        let m = left.shape().shape()[1];
        let k = left.shape().shape()[2];
        let n = right.shape().shape()[2];

        // TODO this could be generalized to consider any possible output permutation
        //   and picking the transpose and storage formats that fit best
        let result = if batch_first {
            self.alloc_tensor_shared(ConcreteShape::new(vec![batch_size, m, n]))
        } else {
            self.alloc_tensor_shared(ConcreteShape::new(vec![m, batch_size, n]))
                .permute(&[1, 0, 2])
        };

        // to ensure we (usually) get a simple strided output, we actually compute
        //   result^T = right^T * left^T
        let args = BatchedMatMulArgs {
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0,
            beta: 0.0,
            a: right.permute(&[0, 2, 1]).to_mat_mul_arg(),
            b: left.permute(&[0, 2, 1]).to_mat_mul_arg(),
            c: result.permute(&[0, 2, 1]).to_mat_mul_arg(),
            batch_count: batch_size as i32,
        };

        self.push(PlanStep::MatMul(args), &result_info.debug_id);

        Ok(result)
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

    fn push(&mut self, step: PlanStep, id: &str) {
        let step_info = StepInfo {
            debug_id: id.to_owned(),
            step,
        };
        self.steps.push(step_info);
    }

    fn visit_fused_conv(&mut self, value: Value) -> VisitResult<Option<PlanTensor>> {
        if self.graph[value].shape.rank() != 4 {
            return Ok(None);
        }

        let mut curr = value;
        let graph = self.graph;

        // relu(curr)?
        let act_mode = if let &Operation::Binary {
            left,
            right,
            op: BinaryOp::Max,
        } = &graph[curr].operation
        {
            let relu_other = if self.can_fuse(left) && self.can_fuse(right) {
                if graph.is_const_filled_with(left, 0.0) {
                    right
                } else if graph.is_const_filled_with(right, 0.0) {
                    left
                } else {
                    return Ok(None);
                }
            } else {
                return Ok(None);
            };

            curr = relu_other;
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
            if !self.can_fuse(left) {
                return Ok(None);
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
                    return Ok(None);
                }
            } else if is_res {
                // has to be res
                if res.is_none() {
                    res = Some(right);
                } else {
                    return Ok(None);
                }
            } else {
                return Ok(None);
            }

            curr = left;
        }

        if let &Operation::Conv { input, filter, details } = &graph[curr].operation {
            // collect input tensors
            if let Some(res) = res {
                if graph[input].shape != graph[res].shape {
                    // rejecting the conv here is fine,
                    //   since it will just be retried later without the following res addition
                    return Ok(None);
                }
            }

            // TODO for conv the input & res need simple strides, what about filter and bias?
            let debug_id = &self.graph[value].debug_id;

            let input = self.visit_ensure_simple_strides(input, &format!("{}_input", debug_id))?;
            let filter = self.visit(filter)?;
            let bias = bias.map(|bias| self.visit(bias)).transpose()?;
            let res = res
                .map(|res| self.visit_ensure_simple_strides(res, &format!("{}_res", debug_id)))
                .transpose()?;

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
            self.push(PlanStep::Conv(args), &graph[value].debug_id);

            Ok(Some(output))
        } else {
            Ok(None)
        }
    }

    fn plan_copy_tensor(&mut self, old: &PlanTensor, new: &PlanTensor, id: &str) {
        assert_eq!(old.shape().shape(), new.shape().shape());

        self.plan_scalar_op("*x0 = *x1;", vec![new.clone(), old.clone()], id);
    }

    fn plan_scalar_op(&mut self, operation: &str, operands: Vec<PlanTensor>, id: &str) {
        // add extra axis since ironically the scalar kernel doesn't work for scalar operands
        let operands = if operands[0].shape().rank() == 0 {
            operands.into_iter().map(|op| op.view(vec![1]).unwrap()).collect_vec()
        } else {
            operands
        };

        let shapes = operands.iter().map(|operand| operand.shape().clone()).collect_vec();
        let kernel = ScalarKernel::new_for_shapes(self.device(), operation, &shapes);

        let args = ScalarOpArgs { kernel, operands };
        self.push(PlanStep::ScalarOp(args), id);
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

    fn can_fuse(&self, value: Value) -> bool {
        self.graph.is_hidden_with_users(value, 1)
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
