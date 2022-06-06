use std::fmt::{Debug, Display, Formatter};
use std::time::Instant;

use itertools::zip;

use cuda_sys::wrapper::handle::{CublasHandle, CudaStream, CudnnHandle, Device};
use cuda_sys::wrapper::mem::device::DevicePtr;
use cuda_sys::wrapper::status::Status;
use nn_graph::graph::Graph;

use crate::device_tensor::DeviceTensor;
use crate::kernels;
use crate::planner::{MemoryUsage, Plan, Planner};
use crate::step::{GatherArgs, LayernormOpArgs, ReduceOpArgs, ScalarOpArgs, SoftmaxOpArgs, Step, StepInfo};
use crate::util::debug_vec_multiline;

pub struct CudaExecutor {
    pub handles: Handles,

    pub inputs: Vec<DeviceTensor>,
    pub outputs: Vec<DeviceTensor>,

    pub batch_size: usize,
    pub mem_usage: MemoryUsage,
    steps: Vec<StepInfo<DevicePtr>>,

    profile: bool,
    last_profile: Option<Profile>,

    output_buffers: Option<Vec<Vec<f32>>>,
}

#[derive(Debug)]
pub struct Handles {
    pub cudnn: CudnnHandle,
    pub cublas: CublasHandle,
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
    pub gather: f32,

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

        CudaExecutor {
            handles,
            batch_size,
            mem_usage,
            inputs,
            outputs,
            steps,
            profile: false,
            last_profile: None,
            output_buffers: None,
        }
    }

    pub fn stream(&self) -> &CudaStream {
        self.handles.stream()
    }

    pub fn evaluate(&mut self, inputs: &[&[f32]]) -> &[Vec<f32>] {
        assert_eq!(inputs.len(), self.inputs.len());

        unsafe {
            // copy inputs
            self.stream().synchronize();
            for (tensor, buffer) in zip(&self.inputs, inputs) {
                tensor.copy_from_host_staged(buffer);
            }

            // run the steps
            self.stream().synchronize();
            self.run_async();
            self.stream().synchronize();

            // initialize output buffers if this is the first time we need them
            let outputs = &self.outputs;
            let output_buffers = self.output_buffers.get_or_insert_with(|| {
                outputs
                    .iter()
                    .map(|tensor| vec![f32::NAN; tensor.shape().size()])
                    .collect()
            });

            // copy outputs
            assert_eq!(output_buffers.len(), self.outputs.len());
            for (tensor, buffer) in zip(&self.outputs, output_buffers) {
                tensor.copy_to_host_staged(buffer);
            }
            self.handles.stream().synchronize();

            // cannot be None, we just initialized this
            self.output_buffers.as_ref().unwrap()
        }
    }

    /// Run the steps in this executor. Does no explicit before/after synchronization,
    /// so ensure inputs are written and synchronize before reading outputs.
    pub unsafe fn run_async(&mut self) {
        assert_eq!(Device::current(), self.stream().device());

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
                    Step::Gather { .. } => &mut profile.gather,
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

impl Step<DevicePtr> {
    unsafe fn run(&self, handles: &Handles) {
        match self {
            Step::Conv(args) => {
                args.run(&handles.cudnn);
            }
            Step::MatMul(args) => {
                // schedule blas wait for cudnn
                let cuda_event = handles.cudnn.stream().record_event();
                handles.cublas.stream().wait_for_event(&cuda_event);

                // schedule operation on blas
                args.run(&handles.cublas);

                // schedule cudnn wait for blas
                let blas_event = handles.cublas.stream().record_event();
                handles.cudnn.stream().wait_for_event(&blas_event);
            }
            Step::ScalarOp(ScalarOpArgs { kernel, operands }) => {
                kernel.run(handles.cudnn.stream(), operands);
            }
            Step::ReduceOp(ReduceOpArgs { kernel, input, output }) => kernel.run(handles.cudnn.stream(), input, output),
            Step::SoftmaxOp(SoftmaxOpArgs { kernel, input, output }) => {
                kernel.run(handles.cudnn.stream(), input, output)
            }
            Step::LayernormOp(LayernormOpArgs {
                kernel,
                input0,
                input1,
                output,
            }) => kernel.run(handles.cudnn.stream(), input0, input1.as_ref(), output),
            Step::Gather(GatherArgs {
                input,
                axis,
                indices,
                output,
            }) => {
                assert!(
                    *axis == 1 && input.shape().rank() == 2,
                    "Gather only supported for rank 2 input and axis 1, got shape {:?} and axis {}",
                    input.shape(),
                    axis
                );
                assert!(
                    indices.shape().rank() == 1 && indices.shape().has_simple_strides(),
                    "Gather indices must be rank-1 tensor with simple strides",
                );

                kernels::gather2dAxis1FloatFloat(
                    handles.cudnn.stream().inner(),
                    input.shape().shape()[0] as i32,
                    input.shape().shape()[1] as i32,
                    input.shape().strides()[0] as i32,
                    input.shape().strides()[1] as i32,
                    indices.shape().size() as i32,
                    input.ptr().ptr() as *const f32,
                    indices.ptr().ptr() as *const f32,
                    output.ptr().ptr() as *mut f32,
                )
                .unwrap();
            }
        }
    }
}

impl Handles {
    pub fn device(&self) -> Device {
        self.stream().device()
    }

    pub fn stream(&self) -> &CudaStream {
        self.cudnn.stream()
    }
}

impl Debug for CudaExecutor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CudaExecutor {{")?;

        writeln!(f, "    batch_size: {},", self.batch_size)?;
        writeln!(f, "    mem_usage: {:?},", self.mem_usage)?;
        writeln!(f, "    profile: {},", self.profile)?;

        writeln!(f, "    inputs: {:?},", debug_vec_multiline("    ", &self.inputs))?;
        writeln!(f, "    outputs: {:?},", debug_vec_multiline("    ", &self.outputs))?;
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

        let total = self.conv + self.mat_mul + self.scalar_op + self.softmax_op + self.gather;
        let mut line = |name, time| writeln!(f, "  {} {:>10.4} ms  {:>4.2}", name, time * 1e3, time / total);

        line("Conv:      ", self.conv)?;
        line("Matmul:    ", self.mat_mul)?;
        line("Scalar:    ", self.scalar_op)?;
        line("Softmax:   ", self.softmax_op)?;
        line("Layernorm: ", self.layernorm_op)?;
        line("Gather:    ", self.gather)?;

        writeln!(f, "  ==============================")?;
        writeln!(f, "  Total GPU:  {:>10.4} ms", self.total_gpu * 1e3)?;
        writeln!(f, "  Total CPU:  {:>10.4} ms", self.total_cpu * 1e3)?;
        writeln!(f, "  Overhead:   {:>10.4} ms", self.timing_overhead * 1e3)?;

        writeln!(f, "}}")?;

        Ok(())
    }
}
