use std::fmt::{Debug, Display, Formatter};
use std::time::Instant;

use bytemuck::{cast_slice, Pod};
use bytemuck::checked::cast_slice_mut;
use itertools::{multizip, zip_eq};

use kn_cuda_sys::wrapper::handle::{CublasHandle, CudaStream, CudnnHandle, Device};
use kn_cuda_sys::wrapper::mem::device::DevicePtr;
use kn_cuda_sys::wrapper::mem::pinned::PinnedMem;
use kn_graph::{dispatch_dtensor, dispatch_dtype};
use kn_graph::dtype::{DBool, DTensor, Tensor};
use kn_graph::graph::Graph;

use crate::device_tensor::DeviceTensor;
use crate::planner::{MemoryUsage, Plan, Planner};
use crate::step::{Handles, Step, StepInfo};
use crate::util::debug_vec_multiline;

pub struct CudaExecutor {
    pub handles: Handles,

    pub device_inputs: Vec<DeviceTensor>,
    pub device_outputs: Vec<DeviceTensor>,

    pub batch_size: usize,
    pub mem_usage: MemoryUsage,
    steps: Vec<StepInfo<DevicePtr>>,

    profile: bool,
    last_profile: Option<Profile>,

    // TODO switch to single in/out buffer each, so we do a single memcpy between host and device?
    buffer_inputs: Vec<PinnedMem>,
    buffer_outputs: Vec<PinnedMem>,
    tensor_outputs: Vec<DTensor>,
}

#[derive(Default, Debug, Clone)]
pub struct Profile {
    pub steps: Vec<String>,

    pub conv: f32,
    pub mat_mul: f32,
    pub scalar_op: f32,
    pub reduce_op: f32,
    pub softmax_op: f32,
    pub layernorm_op: f32,
    pub gather_op: f32,

    pub total_cpu: f32,
    pub total_gpu: f32,
    pub timing_overhead: f32,
}

impl CudaExecutor {
    pub fn new(device: Device, graph: &Graph, batch_size: usize) -> Self {
        let handles = Handles {
            cudnn: CudnnHandle::new(device),
            cublas: CublasHandle::new(device),
        };

        let Plan {
            inputs,
            outputs,
            steps,
            mem_usage,
        } = Planner::plan(&handles, graph, batch_size);

        let buffer_inputs = inputs
            .iter()
            .map(|x| {
                let len_bytes = x.strided_shape().size() * x.dtype().size().bytes();
                PinnedMem::alloc(len_bytes, false)
            })
            .collect();
        let buffer_outputs = outputs
            .iter()
            .map(|x| {
                let len_bytes = x.strided_shape().size() * x.dtype().size().bytes();
                PinnedMem::alloc(len_bytes, false)
            })
            .collect();
        let tensor_outputs = outputs
            .iter()
            .map(|x| {
                let shape = x.strided_shape().shape().to_vec();
                let dtype = x.dtype();
                dispatch_dtype!(dtype, |_T, _fs, ft| ft(Tensor::zeros(shape)))
            })
            .collect();

        CudaExecutor {
            handles,
            batch_size,
            mem_usage,
            device_inputs: inputs,
            device_outputs: outputs,
            steps,
            profile: false,
            last_profile: None,
            buffer_inputs,
            buffer_outputs,
            tensor_outputs,
        }
    }

    pub fn stream(&self) -> &CudaStream {
        self.handles.stream()
    }

    // TODO accept views as inputs? introduce DView struct/alias?
    pub fn evaluate(&mut self, inputs: &[DTensor]) -> &[DTensor] {
        assert_eq!(inputs.len(), self.device_inputs.len(), "Wrong input count");
        for (i, (input, tensor)) in zip_eq(inputs, &self.device_inputs).enumerate() {
            assert_eq!(input.shape(), tensor.strided_shape().shape(), "Wrong shape for input {}", i);
            assert_eq!(input.dtype(), tensor.dtype(), "Wrong dtype for input {}", i);
        }

        unsafe {
            // make sure nothing else is using the buffers
            self.stream().synchronize();

            for (input, buffer, tensor) in multizip((inputs, &self.buffer_inputs, &self.device_inputs)) {
                // copy inputs to buffer
                // TODO is there a simple way to avoid the potential extra layout copy?
                dispatch_dtensor!(input, |T, _f, input| {
                    let input = input.as_standard_layout();
                    let input_slice = input.as_slice().unwrap();
                    buffer.as_slice().copy_from_slice(cast_slice::<T, u8>(input_slice));
                });

                // copy buffer to device
                assert!(tensor.strided_shape().has_simple_strides());
                tensor.ptr().copy_linear_from_host_async(buffer, self.stream());
            }

            // run the steps
            self.run_async();

            // copy outputs to buffers
            for (buffer, tensor) in zip_eq(&mut self.buffer_outputs, &self.device_outputs) {
                tensor.ptr().copy_linear_to_host_async(buffer, self.handles.stream());
            }

            // wait for everything to complete
            self.stream().synchronize();

            // copy buffers to tensors
            // TODO interleave this with copying to host?
            for (buffer, tensor) in zip_eq(&self.buffer_outputs, &mut self.tensor_outputs) {
                let buffer: &[u8] = buffer.as_slice();

                unsafe fn branch<T: Pod>(tensor: &mut Tensor<T>, buffer: &[u8]) {
                    cast_slice_mut::<T, u8>(tensor.as_slice_mut().unwrap()).copy_from_slice(buffer);
                }

                match tensor {
                    DTensor::F32(tensor) => branch::<f32>(tensor, buffer),
                    DTensor::F64(tensor) => branch::<f64>(tensor, buffer),
                    DTensor::I8(tensor) => branch::<i8>(tensor, buffer),
                    DTensor::I16(tensor) => branch::<i16>(tensor, buffer),
                    DTensor::I32(tensor) => branch::<i32>(tensor, buffer),
                    DTensor::I64(tensor) => branch::<i64>(tensor, buffer),
                    DTensor::U8(tensor) => branch::<u8>(tensor, buffer),
                    DTensor::U16(tensor) => branch::<u16>(tensor, buffer),
                    DTensor::U32(tensor) => branch::<u32>(tensor, buffer),
                    DTensor::U64(tensor) => branch::<u64>(tensor, buffer),

                    // do a manual copy, with proper error checking
                    // we can't use bytemuck here since it rightfully doesn't want to cast &mut bool/DBool to &mut u8
                    DTensor::Bool(tensor) => {
                        let mut fail = false;
                        for (i, x) in tensor.iter_mut().enumerate() {
                            let y = buffer[i];
                            *x = DBool(y != 0);
                            fail |= y > 1;
                        }
                        assert!(!fail);
                    }
                }
            }
        }

        &self.tensor_outputs
    }

    /// Run the steps in this executor. Does no explicit before/after synchronization,
    /// so ensure inputs are written and synchronize before reading outputs.
    pub unsafe fn run_async(&mut self) {
        assert_eq!(self.stream().device(), Device::current());

        if !self.profile {
            for step_info in &self.steps {
                step_info.step.run(&self.handles);
            }

            self.last_profile = None
        } else {
            let mut timers = vec![];

            let start_gpu = self.stream().record_event();
            start_gpu.synchronize();

            let start_cpu = Instant::now();

            for step_info in &self.steps {
                let start = self.stream().record_event();
                step_info.step.run(&self.handles);
                let end = self.stream().record_event();

                if self.profile {
                    timers.push((step_info, start, end));
                }
            }

            let end_gpu = self.stream().record_event();
            self.stream().synchronize();

            let end_cpu = Instant::now();

            let mut profile = Profile::default();

            for (i, (step_info, start, end)) in timers.iter().enumerate() {
                let time = end.time_elapsed_since(start);

                *match step_info.step {
                    Step::Conv { .. } => &mut profile.conv,
                    Step::MatMul { .. } => &mut profile.mat_mul,
                    Step::ScalarOp { .. } => &mut profile.scalar_op,
                    Step::ReduceOp { .. } => &mut profile.reduce_op,
                    Step::SoftmaxOp { .. } => &mut profile.softmax_op,
                    Step::LayernormOp { .. } => &mut profile.layernorm_op,
                    Step::GatherOp { .. } => &mut profile.gather_op,
                } += time;

                profile
                    .steps
                    .push(format!("{: >4} time {:>10.4} ms, step {:?}", i, time * 1e3, step_info));
            }

            let overhead_end = Instant::now();
            profile.total_gpu = end_gpu.time_elapsed_since(&start_gpu);
            profile.total_cpu = (end_cpu - start_cpu).as_secs_f32();
            profile.timing_overhead = (overhead_end - end_cpu).as_secs_f32();

            self.last_profile = Some(profile)
        }
    }

    pub fn set_profile(&mut self, profile: bool) {
        self.profile = profile;
    }

    pub fn last_profile(&self) -> Option<&Profile> {
        self.last_profile.as_ref()
    }
}

impl Debug for CudaExecutor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CudaExecutor {{")?;

        writeln!(f, "    batch_size: {},", self.batch_size)?;
        writeln!(f, "    mem_usage: {:?},", self.mem_usage)?;
        writeln!(f, "    profile: {},", self.profile)?;

        writeln!(f, "    inputs: {:?},", debug_vec_multiline("    ", &self.device_inputs))?;
        writeln!(f, "    outputs: {:?},", debug_vec_multiline("    ", &self.device_outputs))?;
        writeln!(f, "    steps: {:?},", debug_vec_multiline("    ", &self.steps))?;

        writeln!(f, "}}")?;

        Ok(())
    }
}

impl Display for Profile {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Profile {{\n  steps: [\n")?;
        for step in &self.steps {
            writeln!(f, "    {}", step)?;
        }
        write!(f, "  ]\n\n")?;

        let total = self.conv
            + self.mat_mul
            + self.scalar_op
            + self.reduce_op
            + self.softmax_op
            + self.layernorm_op
            + self.gather_op;
        let mut line = |name, time| writeln!(f, "  {} {:>10.4} ms  {:>4.2}", name, time * 1e3, time / total);

        line("Conv:      ", self.conv)?;
        line("Matmul:    ", self.mat_mul)?;
        line("Scalar:    ", self.scalar_op)?;
        line("Reduce:    ", self.reduce_op)?;
        line("Softmax:   ", self.softmax_op)?;
        line("Layernorm: ", self.layernorm_op)?;
        line("Gather:    ", self.gather_op)?;

        writeln!(f, "  ==============================")?;
        writeln!(f, "  Total GPU:  {:>10.4} ms", self.total_gpu * 1e3)?;
        writeln!(f, "  Total CPU:  {:>10.4} ms", self.total_cpu * 1e3)?;
        writeln!(f, "  Overhead:   {:>10.4} ms", self.timing_overhead * 1e3)?;

        writeln!(f, "}}")?;

        Ok(())
    }
}
