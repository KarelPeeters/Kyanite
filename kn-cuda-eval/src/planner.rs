use std::cmp::{max, min};
use std::collections::HashMap;
use std::fmt::Write;

use bytemuck::cast_slice;
use internal_iterator::InternalIterator;
use itertools::Itertools;

use kn_cuda_sys::bindings::{cudnnActivationMode_t, cudnnDataType_t};
use kn_cuda_sys::wrapper::descriptor::{ActivationDescriptor, ConvolutionDescriptor};
use kn_cuda_sys::wrapper::group::{BatchedMatMulArgs, FusedConvolutionArgs};
use kn_cuda_sys::wrapper::handle::Device;
use kn_cuda_sys::wrapper::mem::device::DevicePtr;
use kn_cuda_sys::wrapper::operation::STANDARD_CONV_ALGO;
use kn_graph::dispatch_dtensor;
use kn_graph::dtype::{DisplayCFloat, DScalar, DType};
use kn_graph::graph::{BinaryOp, Graph, Operation, SliceRange, UnaryOp, Value};
use kn_graph::optimizer::recurse::heap_recurse;
use kn_graph::shape::{ConcreteShape, Size};

use crate::autokernel::gather::GatherKernel;
use crate::autokernel::layernorm::LayernormKernel;
use crate::autokernel::reduce::{ReduceCode, ReduceKernel};
use crate::autokernel::scalar::ScalarKernel;
use crate::autokernel::softmax::SoftmaxKernel;
use crate::device_tensor::DeviceTensor;
use crate::offset_tensor::{OffsetPtr, PtrTensor};
use crate::shape::StridedShape;
use crate::step::{GatherOpArgs, Handles, LayernormOpArgs, ReduceOpArgs, ScalarOpArgs, SoftmaxOpArgs, Step, StepInfo};

/// Planner converts a Graph into a concrete cuda execution plan.
///
/// This planner is implemented fully recursively, but this means that we can hit stack-overflows for deep graphs.
/// To solve this we "break" the recursion using the `VisitResult<T>` type:
/// * `Ok(T)` means the mapping was completed.
/// * `Err(other_value)` means `other_value` should be visited first, after which the original value should be tried again.
///
/// This has a few consequences:
/// * All operands should be visited before
///     * any step is pushed
///     * any dedicated tensor is allocated
///     * any other non-idempotent state is modified
/// * We might repeatedly "allocate" tensors, but this is fine since dead allocations are skipped later.
pub(crate) struct Planner<'a> {
    handles: &'a Handles,
    graph: &'a Graph,
    batch_size: usize,

    shared_buffers: Vec<SharedBufferInfo>,
    max_zero_bytes: usize,

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
    /// Bytes allocated for dedicated tensors (weights)
    pub dedicated_bytes: usize,
    /// Bytes allocated for shared tensors (inputs, outputs, hidden states)
    pub shared_bytes: usize,
    /// Bytes allocated for fixed-zero tensors.
    pub zero_bytes: usize,

    /// Bytes that are theoretically necessary for the values that are live at the same time.
    /// This is a lower bound for `shared_bytes`.
    pub hypo_shared_bytes_peak: usize,
    /// Bytes that would be necessary if each shared buffer got a distinct allocation, without any reuse.
    pub hypo_shared_bytes_total: usize,
}

#[derive(Debug)]
struct SharedBufferInfo {
    size_bytes: usize,
    #[allow(dead_code)]
    debug_value: Option<Value>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum PlanBuffer {
    Dedicated(DevicePtr),
    Shared { index: usize },
    Zero,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct PlanPtr {
    debug_value: Option<Value>,
    buffer: PlanBuffer,
    offset_bytes: isize,
}

type PlanTensor = PtrTensor<PlanPtr>;
type PlanStep = Step<PlanPtr>;
type ExecStep = Step<DevicePtr>;
type PlanStepInfo = StepInfo<PlanPtr>;
type ExecStepInfo = StepInfo<DevicePtr>;

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
            .map(|&output| planner.visit_completely_ensure_simple_strides(output))
            .collect_vec();

        let buffer_count = planner.shared_buffers.len();
        let step_count = planner.steps.len();

        // determine live ranges for shared tensors
        let live_ranges = {
            let mut live_ranges = vec![(step_count, 0); buffer_count];

            for (si, step_info) in planner.steps.iter().enumerate() {
                step_info.step.ptr_operands().for_each(|op| {
                    if let &PlanBuffer::Shared { index } = &op.value.buffer {
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

        let mut shared_bytes = 0;
        let mut curr_shared_bytes = 0;
        let mut hypo_shared_bytes_peak = 0;
        let mut hypo_shared_bytes_total = 0;

        for si in 0..step_count {
            for (ti, &(start, _)) in live_ranges.iter().enumerate() {
                if start == si {
                    // allocate the given tensor
                    let size_bytes = planner.shared_buffers[ti].size_bytes;
                    curr_shared_bytes += size_bytes;
                    hypo_shared_bytes_peak = max(hypo_shared_bytes_peak, curr_shared_bytes);
                    hypo_shared_bytes_total += size_bytes;

                    let vec = free_allocations.entry(size_bytes).or_insert_with(Vec::new);
                    let ptr = vec.pop().unwrap_or_else(|| {
                        shared_bytes += size_bytes;
                        device.alloc(size_bytes)
                    });
                    assert!(shared_allocations[ti].is_none());
                    shared_allocations[ti] = Some(ptr);
                }
            }

            for (ti, &(start, end)) in live_ranges.iter().enumerate() {
                if start <= si && end == si {
                    // free the given tensor
                    let size_bytes = planner.shared_buffers[ti].size_bytes;
                    curr_shared_bytes -= size_bytes;

                    let ptr = shared_allocations[ti].as_ref().unwrap().clone();
                    let vec = free_allocations.get_mut(&size_bytes).unwrap();
                    vec.push(ptr);
                }
            }
        }

        // collect shared allocations
        let shared_allocations = shared_allocations
            .into_iter()
            .enumerate()
            .map(|(i, ptr)| {
                ptr.or_else(|| {
                    let (start, end) = live_ranges[i];

                    if start <= end {
                        // allocate leftover buffers that are never used before the end (eg. empty concat output tensors)
                        let size_bytes = planner.shared_buffers[i].size_bytes;
                        shared_bytes += size_bytes;
                        Some(device.alloc(size_bytes))
                    } else {
                        // don't allocate buffers that are actually never used
                        None
                    }
                })
            })
            .collect_vec();

        // allocate a single shared zero tensor
        // it's shared between all types, this happens to work even for floats
        let zero_bytes = planner.max_zero_bytes;
        let zero_allocation = device.alloc(zero_bytes);
        unsafe {
            zero_allocation.copy_linear_from_host(&vec![0; zero_bytes]);
        }

        let mem_usage = MemoryUsage {
            dedicated_bytes: planner.dedicated_bytes,
            shared_bytes,
            zero_bytes,
            hypo_shared_bytes_peak,
            hypo_shared_bytes_total,
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
                .map(
                    |PlanStepInfo {
                         debug_value,
                         debug_id,
                         step,
                     }| ExecStepInfo {
                        debug_value,
                        debug_id,
                        step: ctx.realize_step(step),
                    },
                )
                .collect_vec(),
            mem_usage,
        }
    }

    fn new(handles: &'a Handles, graph: &'a Graph, batch_size: usize) -> Self {
        Planner {
            handles,
            graph,
            batch_size,
            shared_buffers: vec![],
            map: Default::default(),
            steps: vec![],
            max_zero_bytes: 0,
            dedicated_bytes: 0,
        }
    }

    fn visit_completely(&mut self, value: Value) -> PlanTensor {
        heap_recurse(value, |value| self.visit_single_cached(value))
    }

    fn visit_completely_ensure_simple_strides(&mut self, value: Value) -> PlanTensor {
        self.visit_completely(value);
        self.visit_ensure_simple_strides(value).unwrap()
    }

    fn visit(&mut self, value: Value) -> Result<PlanTensor, Value> {
        if let Some(result) = self.map.get(&value) {
            return Ok(result.clone());
        }
        return Err(value);
    }

    fn visit_single_cached(&mut self, value: Value) -> VisitResult<PlanTensor> {
        if let Some(result) = self.map.get(&value) {
            return Ok(result.clone());
        }

        let result = self.visit_single_new(value)?;

        // check that the result matches the expected shape and type
        let info = &self.graph[value];
        assert_eq!(result.strided_shape().shape(), info.shape.eval(self.batch_size).dims);
        assert_eq!(result.dtype(), info.dtype);

        self.insert_mapping(value, result.clone());

        Ok(result)
    }

    fn visit_single_new(&mut self, value: Value) -> VisitResult<PlanTensor> {
        if let Some(result) = self.try_visit_fused_conv(value)? {
            return Ok(result);
        }

        let result_info = &self.graph[value];
        let result_shape = result_info.shape.eval(self.batch_size);
        let result_dtype = result_info.dtype;

        let result: PlanTensor = match &result_info.operation {
            &Operation::Input { index: _ } => self.alloc_tensor_shared(result_shape, result_dtype, Some(value)),
            Operation::Constant { tensor } => {
                let result = self.alloc_tensor_dedicated(result_shape, tensor.dtype());

                // copy values
                dispatch_dtensor!(tensor, |T, _f, inner| {
                    let inner = inner.as_standard_layout();
                    let bytes = cast_slice::<T, u8>(inner.as_slice().unwrap());
                    unsafe {
                        result.copy_simple_from_host(bytes);
                    }
                });

                result.map_ptr(|device_ptr| PlanPtr::from_device_ptr(Some(value), device_ptr))
            }
            &Operation::View { input } => {
                let input_tensor = self.visit(input)?;

                // try simple view operation, otherwise restride to dense and then copy
                // TODO if we want the output to be simple strided immediately we need separate input/output shapes
                //    in the scalar kernel
                input_tensor.view(result_shape.dims.clone()).unwrap_or_else(|_| {
                    let input_shape = ConcreteShape::new(input_tensor.strided_shape().shape().to_vec());
                    let result = self.alloc_tensor_shared(input_shape, input_tensor.dtype(), Some(value));
                    self.plan_copy_tensor(&input_tensor, &result, value);
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
                let output = self.alloc_tensor_shared(result_shape, result_dtype, Some(value));

                assert_eq!(input.dtype(), output.dtype());

                // TODO make sure all autokernels support 64-bit memory offsets
                //   and use the proper signedness for all operands and internal variables
                let kernel = GatherKernel::new(
                    self.device(),
                    input.strided_shape(),
                    indices.strided_shape(),
                    output.strided_shape(),
                    input.dtype(),
                    indices.dtype(),
                    axis,
                );

                let args = GatherOpArgs {
                    kernel,
                    input,
                    indices,
                    output: output.clone(),
                };

                self.push(PlanStep::GatherOp(args), value);
                output
            }
            &Operation::Concat { ref inputs, axis } => {
                let result = self.alloc_tensor_shared(result_shape, result_dtype, Some(value));
                let inputs: Vec<PlanTensor> = inputs.iter().map(|&x| self.visit(x)).try_collect()?;

                // copy each input into the corresponding slice of the output
                let mut curr_start = 0;

                for input in inputs {
                    let curr_size = input.strided_shape().shape()[axis];
                    let curr_range = SliceRange::simple(curr_start, curr_start + curr_size);

                    let curr_result = result.slice(axis, curr_range);
                    self.plan_copy_tensor(&input, &curr_result, value);

                    curr_start += curr_size;
                }

                result
            }
            &Operation::Conv { .. } => {
                unreachable!("Conv should have been handled by conv fuser")
            }
            &Operation::MatMul { left, right } => self.visit_matmul(value, left, right, true)?,
            &Operation::Unary { .. } | &Operation::Binary { .. } => self.visit_fused_scalar(value)?,
            &Operation::Softmax { input, axis } => {
                assert_eq!(result_dtype, DType::F32);

                let (input_scale, input) = self.visit_scalable_value(input)?;
                let input_scale = input_scale.unwrap_f32().unwrap();

                let output = self.alloc_tensor_shared(result_shape, result_dtype, Some(value));

                let kernel = SoftmaxKernel::new(
                    self.device(),
                    input.strided_shape(),
                    output.strided_shape(),
                    axis,
                    input_scale,
                );

                let args = SoftmaxOpArgs {
                    kernel,
                    input,
                    output: output.clone(),
                };

                self.push(PlanStep::SoftmaxOp(args), value);
                output
            }
            &Operation::Layernorm { input, axis, eps } => {
                assert_eq!(result_dtype, DType::F32);

                let (alpha_0, input_0, alpha_1, input_1) = self.visit_scalable_added_pair(input)?;
                let alpha_0 = alpha_0.unwrap_f32().unwrap();
                let alpha_1 = alpha_1.unwrap_f32().unwrap();

                if let Some(input_1) = &input_1 {
                    assert_eq!(input_0.strided_shape(), input_1.strided_shape());
                }

                let output = self.alloc_tensor_shared(result_shape, result_dtype, Some(value));

                let kernel = LayernormKernel::new(
                    self.device(),
                    input_0.strided_shape(),
                    output.strided_shape(),
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

                self.push(PlanStep::LayernormOp(args), value);
                output
            }
            &Operation::Reduce { input, ref axes, op } => {
                let result_size = result_shape.size();
                let dtype = result_dtype;

                let input = self.visit(input)?;
                let output = self.alloc_tensor_shared(result_shape, result_dtype, Some(value));

                let identity = op.identity();
                let (operation, is_mean) = op.operation();

                let post_process = if is_mean {
                    assert_eq!(dtype, DType::F32);
                    let scale = (result_size as f64 / input.strided_shape().size() as f64) as f32;
                    format!("curr * {}", DisplayCFloat(scale))
                } else {
                    "curr".to_owned()
                };

                let code = ReduceCode {
                    ty: dtype.as_c_str().to_owned(),
                    identity: format!("{}", DisplayCFloat(identity)),
                    operation: binary_op_str(operation, "curr", "x"),
                    post_process,
                };

                let kernel =
                    ReduceKernel::new(self.device(), code, input.strided_shape(), output.strided_shape(), axes);

                let args = ReduceOpArgs {
                    kernel,
                    input,
                    output: output.clone(),
                };
                self.push(PlanStep::ReduceOp(args), value);

                output
            }
        };

        Ok(result)
    }

    /// Returns values of the form `alpha_0 * input_0 + alpha_1 * input_1`
    ///
    /// `None` for an input means that input is not present and should be treated as zero.
    fn visit_scalable_added_pair(&mut self, input: Value) -> VisitResult<(DScalar, PlanTensor, DScalar, Option<PlanTensor>)> {
        let dtype = self.graph[input].dtype;
        let zero = dtype.specials().zero;
        let one = dtype.specials().one;

        let result = if let &Operation::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } = &self.graph[input].operation
        {
            let (alpha_0, input_0) = self.visit_scalable_value(left)?;
            let (alpha_1, input_1) = self.visit_scalable_value(right)?;

            if input_0.strided_shape() != input_1.strided_shape() {
                // fallback to scalar operation
                let total = self.visit(input)?;
                (one, total, zero, None)
            } else {
                (alpha_0, input_0, alpha_1, Some(input_1))
            }
        } else {
            let (alpha_0, input_0) = self.visit_scalable_value(input)?;
            (alpha_0, input_0, zero, None)
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

    fn visit_ensure_simple_strides(&mut self, value: Value) -> VisitResult<PlanTensor> {
        let inner = self.visit(value)?;

        let result = if inner.strided_shape().has_simple_strides() {
            inner
        } else {
            let shape = ConcreteShape::new(inner.strided_shape().shape().to_vec());
            let new = self.alloc_tensor_shared(shape, inner.dtype(), Some(value));
            self.plan_copy_tensor(&inner, &new, value);
            new
        };

        Ok(result)
    }

    fn visit_scalable_value(&mut self, value: Value) -> VisitResult<(DScalar, PlanTensor)> {
        let dtype = self.graph[value].dtype;
        let one = dtype.specials().one;

        if let &Operation::Binary {
            op: BinaryOp::Mul,
            left,
            right,
        } = &self.graph[value].operation
        {
            // TODO remove this duplication once we have constant canonicalization in graph
            if let Some(alpha) = self.graph.as_single_const(left) {
                return Ok((alpha, self.visit(right)?));
            }
            if let Some(alpha) = self.graph.as_single_const(right) {
                return Ok((alpha, self.visit(left)?));
            }
        }

        Ok((one, self.visit(value)?))
    }

    fn visit_matmul(&mut self, value: Value, left: Value, right: Value, batch_first: bool) -> VisitResult<PlanTensor> {
        // TODO support other dtypes?
        assert_eq!(self.graph[value].dtype, DType::F32);
        assert_eq!(self.graph[left].dtype, self.graph[right].dtype);
        let dtype = self.graph[left].dtype;

        let left = self.visit(left)?;
        let right = self.visit(right)?;

        assert!(left.strided_shape().rank() == 3 && right.strided_shape().rank() == 3);
        let batch_size = left.strided_shape().shape()[0];
        let m = left.strided_shape().shape()[1];
        let k = left.strided_shape().shape()[2];
        let n = right.strided_shape().shape()[2];

        // TODO this could be generalized to consider any possible output permutation
        //   and picking the transpose and storage formats that fit best
        let result = if batch_first {
            self.alloc_tensor_shared(ConcreteShape::new(vec![batch_size, m, n]), dtype, Some(value))
        } else {
            self.alloc_tensor_shared(ConcreteShape::new(vec![m, batch_size, n]), dtype, Some(value))
                .permute(&[1, 0, 2])
        };

        // TODO use alpha for operand scaling?
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

        self.push(PlanStep::MatMul(args), value);

        Ok(result)
    }

    fn insert_mapping(&mut self, value: Value, result: PlanTensor) {
        assert_eq!(
            result.strided_shape().shape(),
            self.graph[value].shape.eval(self.batch_size).dims,
            "Got wrong result shape"
        );
        let prev = self.map.insert(value, result);
        assert!(prev.is_none());
    }

    fn push(&mut self, step: PlanStep, debug_value: Value) {
        let step_info = StepInfo {
            debug_value,
            debug_id: self.graph[debug_value].debug_id.clone(),
            step,
        };
        self.steps.push(step_info);
    }

    fn try_visit_fused_conv(&mut self, value: Value) -> VisitResult<Option<PlanTensor>> {
        if self.graph[value].dtype != DType::F32 || self.graph[value].shape.rank() != 4 {
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
                if graph.is_const_zero(left) {
                    right
                } else if graph.is_const_zero(right) {
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
            assert_eq!(details.dtype, DType::F32);
            let dtype = cudnnDataType_t::CUDNN_DATA_FLOAT;

            // collect input tensors
            if let Some(res) = res {
                if graph[input].shape != graph[res].shape {
                    // rejecting the conv here is fine,
                    //   since it will just be retried later without the following res addition
                    return Ok(None);
                }
            }

            // TODO for conv the input & res need simple strides, what about filter and bias?
            let input = self.visit_ensure_simple_strides(input)?;
            let filter = self.visit(filter)?;
            let bias = bias.map(|bias| self.visit(bias)).transpose()?;
            let res = res.map(|res| self.visit_ensure_simple_strides(res)).transpose()?;

            let bias = bias.unwrap_or_else(|| {
                let bias_shape = ConcreteShape::new(vec![1, details.output_channels, 1, 1]);
                self.alloc_tensor_zero(bias_shape, details.dtype, None)
            });

            let output_shape = graph[curr].shape.eval(self.batch_size);
            let output = self.alloc_tensor_shared(output_shape, details.dtype, Some(value));

            // build descriptors
            assert_eq!(input.dtype(), DType::F32);
            assert_eq!(bias.dtype(), DType::F32);
            assert_eq!(filter.dtype(), DType::F32);
            assert_eq!(output.dtype(), DType::F32);
            let input_desc = input.strided_shape().descriptor(dtype);
            let bias_desc = bias.strided_shape().descriptor(dtype);
            let filter_desc = filter.strided_shape().filter_descriptor(dtype);
            let output_desc = output.strided_shape().descriptor(dtype);

            let conv_desc = ConvolutionDescriptor::new(
                details.padding_y as i32,
                details.padding_x as i32,
                details.stride_y as i32,
                details.stride_x as i32,
                1,
                1,
                dtype,
            );
            let act_desc = ActivationDescriptor::new(act_mode, 0.0);
            let algo = STANDARD_CONV_ALGO;

            assert_eq!(
                &conv_desc.output_shape(&input_desc, &filter_desc).map(|i| i as usize),
                output.strided_shape().shape(),
                "Output shape mismatch between cudnn and graph for value {:?} with operation {:?}",
                curr,
                graph[curr].operation,
            );

            let work_size_bytes =
                conv_desc.workspace_size(&self.handles.cudnn, algo, &input_desc, &filter_desc, &output_desc);
            let work_ptr = self.alloc_buffer_shared(work_size_bytes, None);

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
            self.push(PlanStep::Conv(args), value);

            Ok(Some(output))
        } else {
            Ok(None)
        }
    }

    fn visit_fused_scalar(&mut self, value: Value) -> VisitResult<PlanTensor> {
        // TODO proper scalar fusion:
        //     keep fusing while bandwidth does not increase
        //     is greedy fusion enough or do we need some proper clique finding?
        assert!(matches!(
            self.graph[value].operation,
            Operation::Unary { .. } | Operation::Binary { .. }
        ));

        let result_info = &self.graph[value];
        let result_shape = result_info.shape.eval(self.batch_size);
        let result = self.alloc_tensor_shared(result_shape, result_info.dtype, Some(value));

        let mut block = ScalarBlock::default();
        let result_y = self.visit_fused_scalar_recurse(value, &mut block, true)?;
        block.store_operand_y(&result, result_y);

        self.plan_scalar_op(&block.operation, block.operands, value);

        Ok(result)
    }

    fn visit_fused_scalar_recurse(
        &mut self,
        value: Value,
        block: &mut ScalarBlock,
        is_root: bool,
    ) -> VisitResult<usize> {
        // TODO this should really check whether all users can be fused into a scalar block

        // try fusing conv (and looking in the cache for fused convs) first,
        //   to avoid "stealing" the relu, add, or bias
        // TODO this is pretty hacky, improve how all of this works
        {
            if let Some(result) = self.map.get(&value) {
                return Ok(block.load_operand_y(result));
            }
            if let Some(result) = self.try_visit_fused_conv(value)? {
                let y = block.load_operand_y(&result);
                self.insert_mapping(value, result);
                return Ok(y);
            }
        }

        let value_info = &self.graph[value];
        let op_str = match &value_info.operation {
            &Operation::Unary { op, input } => {
                let y_input = self.visit_fused_scalar_recurse(input, block, false)?;
                unary_op_str(op, &format!("y{}", y_input))
            }
            &Operation::Binary { op, left, right } => {
                let y_left = self.visit_fused_scalar_recurse(left, block, false)?;
                let y_right = self.visit_fused_scalar_recurse(right, block, false)?;
                binary_op_str(op, &format!("y{}", y_left), &format!("y{}", y_right))
            }
            _ => {
                assert!(!is_root);

                let y = if let Some(c) = self.graph.as_single_const(value) {
                    block.define_y(&c.to_c_str())
                } else {
                    block.load_operand_y(&self.visit(value)?)
                };

                return Ok(y);
            }
        };

        let y_output = block.alloc_y();
        writeln!(&mut block.operation, "{} y{} = {};", value_info.dtype.as_c_str(), y_output, op_str).unwrap();
        Ok(y_output)
    }

    // TODO participate in scalar fusion
    fn plan_copy_tensor(&mut self, old: &PlanTensor, new: &PlanTensor, debug_value: Value) {
        assert_eq!(old.strided_shape().shape(), new.strided_shape().shape());

        self.plan_scalar_op("*x0 = *x1;", vec![new.clone(), old.clone()], debug_value);
    }

    fn plan_scalar_op(&mut self, operation: &str, operands: Vec<PlanTensor>, debug_value: Value) {
        // add extra axis since ironically the scalar kernel doesn't work for scalar operands
        let operands = if operands[0].strided_shape().rank() == 0 {
            operands.into_iter().map(|op| op.view(vec![1]).unwrap()).collect_vec()
        } else {
            operands
        };

        let shapes = operands
            .iter()
            .map(|operand| operand.strided_shape().clone())
            .collect_vec();
        let types = operands.iter().map(|operand| operand.dtype().as_c_str().to_string()).collect_vec();
        let kernel = ScalarKernel::new_for_shapes(self.device(), operation, &shapes, types);

        let args = ScalarOpArgs { kernel, operands };
        self.push(PlanStep::ScalarOp(args), debug_value);
    }

    fn alloc_tensor_dedicated(&mut self, shape: ConcreteShape, dtype: DType) -> DeviceTensor {
        self.dedicated_bytes += shape.size() * dtype.size().bytes();
        DeviceTensor::alloc_simple(self.device(), shape.dims, dtype)
    }

    // TODO add skip for zero-sized buffers?
    fn alloc_buffer_shared(&mut self, size_bytes: usize, debug_value: Option<Value>) -> PlanPtr {
        let index = self.shared_buffers.len();
        let info = SharedBufferInfo {
            debug_value,
            size_bytes,
        };
        self.shared_buffers.push(info);
        PlanPtr::from_parts(debug_value, PlanBuffer::Shared { index }, 0)
    }

    fn alloc_tensor_shared(&mut self, shape: ConcreteShape, dtype: DType, debug_value: Option<Value>) -> PlanTensor {
        let buffer = self.alloc_buffer_shared(dtype.size().bytes() * shape.size(), debug_value);
        let shape = StridedShape::new_simple(shape.dims);
        PlanTensor::from_parts(buffer, shape, dtype)
    }

    fn alloc_tensor_zero(&mut self, shape: ConcreteShape, dtype: DType, debug_value: Option<Value>) -> PlanTensor {
        let shape = StridedShape::new_simple(shape.dims);
        let bytes = dtype.size().bytes() * shape.size();
        self.max_zero_bytes = max(self.max_zero_bytes, bytes);
        PlanTensor::from_parts(PlanPtr::from_parts(debug_value, PlanBuffer::Zero, 0), shape, dtype)
    }

    fn device(&self) -> Device {
        self.handles.device()
    }

    fn can_fuse(&self, value: Value) -> bool {
        self.graph.is_hidden_with_uses(value, 1)
    }
}

#[derive(Debug, Default)]
struct ScalarBlock {
    // map operand to x index
    operands: Vec<PlanTensor>,
    // map operand x to y index
    loaded_operands: Vec<Option<usize>>,

    operation: String,
    next_y_index: usize,
}

impl ScalarBlock {
    fn alloc_y(&mut self) -> usize {
        let y = self.next_y_index;
        self.next_y_index += 1;
        y
    }

    fn push_operand_x(&mut self, operand: &PlanTensor) -> usize {
        if let Some(other) = self.operands.get(0) {
            assert_eq!(operand.strided_shape().shape(), other.strided_shape().shape());
        }

        if let Some(x) = self.operands.iter().position(|o| o == operand) {
            x
        } else {
            let x = self.operands.len();
            self.operands.push(operand.clone());
            self.loaded_operands.push(None);
            x
        }
    }

    fn load_operand_y(&mut self, operand: &PlanTensor) -> usize {
        let x = self.push_operand_x(operand);

        if let Some(y) = self.loaded_operands[x] {
            y
        } else {
            let y = self.alloc_y();
            self.loaded_operands[x] = Some(y);
            writeln!(&mut self.operation, "float y{} = *x{};", y, x).unwrap();
            y
        }
    }

    fn define_y(&mut self, value: &str) -> usize {
        let y = self.alloc_y();
        writeln!(&mut self.operation, "float y{} = {};", y, value).unwrap();
        y
    }

    fn store_operand_y(&mut self, operand: &PlanTensor, y: usize) {
        let x = self.push_operand_x(operand);
        writeln!(&mut self.operation, "*x{} = y{};", x, y).unwrap();
    }
}

#[derive(Debug)]
struct RealizationContext {
    shared_allocations: Vec<Option<DevicePtr>>,
    zero_allocation: DevicePtr,
}

impl RealizationContext {
    fn realize_ptr(&self, ptr: PlanPtr) -> DevicePtr {
        let PlanPtr {
            debug_value: _,
            buffer,
            offset_bytes,
        } = ptr;

        let base = match buffer {
            PlanBuffer::Dedicated(buffer) => buffer,
            PlanBuffer::Shared { index } => self.shared_allocations[index]
                .as_ref()
                .unwrap_or_else(|| panic!("Unused shared buffer {index} is actually used"))
                .clone(),
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
    pub fn from_parts(debug_value: Option<Value>, buffer: PlanBuffer, offset_bytes: isize) -> Self {
        Self {
            debug_value,
            buffer,
            offset_bytes,
        }
    }

    pub fn from_device_ptr(debug_value: Option<Value>, device_ptr: DevicePtr) -> Self {
        // device_ptr already has a built-in offset, so we don't need to use the extra one we have here
        Self::from_parts(debug_value, PlanBuffer::Dedicated(device_ptr), 0)
    }
}

impl OffsetPtr for PlanPtr {
    fn offset_bytes(self, offset_bytes: isize) -> Self {
        PlanPtr::from_parts(self.debug_value, self.buffer, self.offset_bytes + offset_bytes)
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

fn unary_op_str(op: UnaryOp, x: &str) -> String {
    match op {
        UnaryOp::Abs => format!("fabs({})", x),
        UnaryOp::Neg => format!("-({})", x),
        UnaryOp::Sin => format!("sin({})", x),
        UnaryOp::Cos => format!("cos({})", x),
        UnaryOp::Exp => format!("exp({})", x),
        UnaryOp::Log => format!("log({})", x),
        UnaryOp::Sqrt => format!("sqrt({})", x),
        UnaryOp::Sigmoid => format!("1.0 / (1.0 + exp(-({})))", x),
        UnaryOp::Tanh => format!("tanh({})", x),
        UnaryOp::Erf => format!("erff({})", x),
        UnaryOp::Mish => format!("({}) * tanh(log(1.0 + exp({})))", x, x),
        UnaryOp::ValueCast(_to) => todo!(),
        UnaryOp::BitCast(_to) => todo!(),
    }
}
