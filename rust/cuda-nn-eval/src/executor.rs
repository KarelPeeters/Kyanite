use std::fmt::{Debug, Display, Formatter};
use std::time::Instant;

use itertools::{zip, Itertools};

use cuda_sys::wrapper::group::{BatchedMatMulArgs, FusedConvolutionArgs, TensorOpArgs};
use cuda_sys::wrapper::handle::{CublasHandle, CudnnHandle, Device};
use cuda_sys::wrapper::status::Status;
use nn_graph::graph::Graph;

use crate::kernels;
use crate::planner::Planner;
use crate::tensor::DeviceTensor;
use crate::util::debug_vec_multiline;

pub struct CudaExecutor {
    pub handles: Handles,

    pub inputs: Vec<DeviceTensor>,
    pub outputs: Vec<DeviceTensor>,

    pub batch_size: usize,
    steps: Vec<Step>,

    profile: bool,
    last_profile: Option<Profile>,

    output_buffers: Option<Vec<Vec<f32>>>,
}

#[derive(Debug)]
pub struct Handles {
    pub cudnn: CudnnHandle,
    pub cublas: CublasHandle,
}

#[derive(Debug)]
pub enum Step {
    Conv {
        args: FusedConvolutionArgs,
    },
    MatMul {
        args: BatchedMatMulArgs,
    },
    TensorOp {
        args: TensorOpArgs,
    },
    Gather {
        input: DeviceTensor,
        axis: usize,
        indices: DeviceTensor,
        output: DeviceTensor,
    },
}

#[derive(Default, Debug, Clone)]
pub struct Profile {
    pub steps: Vec<String>,

    pub conv: f32,
    pub mat_mul: f32,
    pub tensor_op: f32,
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
        let mut planner = Planner::new(&handles, graph, batch_size);

        // allocate inputs
        let inputs = graph.inputs().iter().map(|&input| planner.visit(input)).collect_vec();

        // plan operations and collect outputs
        let outputs = graph
            .outputs()
            .iter()
            .map(|&output| planner.visit_ensure_simple_strides(output))
            .collect_vec();

        let steps = planner.finish();

        CudaExecutor {
            handles,
            steps,
            inputs,
            outputs,
            batch_size,
            profile: false,
            last_profile: None,
            output_buffers: None,
        }
    }

    pub fn evaluate(&mut self, inputs: &[&[f32]]) -> &[Vec<f32>] {
        assert_eq!(inputs.len(), self.inputs.len());

        unsafe {
            // copy inputs
            self.handles.cudnn.stream().synchronize();
            for (tensor, buffer) in zip(&self.inputs, inputs) {
                tensor.copy_from_host_staged(buffer);
            }

            // run the steps
            self.handles.cudnn.stream().synchronize();
            self.run_async();
            self.handles.cudnn.stream().synchronize();

            // initialize output buffers if this is the first time we need them
            let outputs = &self.outputs;
            let output_buffers = self.output_buffers.get_or_insert_with(|| {
                outputs
                    .iter()
                    .map(|tensor| vec![f32::NAN; tensor.shape.size()])
                    .collect()
            });

            // copy outputs
            assert_eq!(output_buffers.len(), self.outputs.len());
            for (tensor, buffer) in zip(&self.outputs, output_buffers) {
                tensor.copy_to_host_staged(buffer);
                self.handles.cudnn.stream().synchronize();
            }

            // cannot be None, we just initialized this
            self.output_buffers.as_ref().unwrap()
        }
    }

    /// Run the steps in this executor. Does no explicit before/after synchronization,
    /// so ensure inputs are written and synchronize before reading outputs.
    pub unsafe fn run_async(&mut self) {
        if !self.profile {
            for step in &self.steps {
                step.run(&self.handles);
            }

            self.last_profile = None
        } else {
            let mut timers = vec![];
            let start_cpu = Instant::now();
            let start_all;
            let end_all;

            start_all = self.handles.cudnn.stream().record_new_event();

            for step in &self.steps {
                let start = self.handles.cudnn.stream().record_new_event();
                step.run(&self.handles);
                let end = self.handles.cudnn.stream().record_new_event();

                if self.profile {
                    timers.push((step, start, end));
                }
            }

            end_all = self.handles.cudnn.stream().record_new_event();
            self.handles.cudnn.stream().synchronize();

            let end_cpu = Instant::now();

            let mut profile = Profile::default();

            for (i, (step, start, end)) in timers.iter().enumerate() {
                let time = end.time_elapsed_since(start);

                *match step {
                    Step::Conv { .. } => &mut profile.conv,
                    Step::MatMul { .. } => &mut profile.mat_mul,
                    Step::TensorOp { .. } => &mut profile.tensor_op,
                    Step::Gather { .. } => &mut profile.gather,
                } += time;

                profile
                    .steps
                    .push(format!("{: >4} time {:>10.4} ms, step {:?}", i, time * 1e3, step));
            }

            let overhead_end = Instant::now();
            profile.total_gpu = end_all.time_elapsed_since(&start_all);
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

impl Step {
    unsafe fn run(&self, handles: &Handles) {
        match self {
            Step::Conv { args } => {
                args.run(&handles.cudnn);
            }
            Step::MatMul { args } => {
                // schedule blas wait for cuda
                let cuda_event = handles.cudnn.stream().record_new_event();
                handles.cublas.stream().wait_for_event(&cuda_event);

                // schedule operation on blas
                args.run(&handles.cublas);

                // schedule cuda wait for blas
                let blas_event = handles.cublas.stream().record_new_event();
                handles.cudnn.stream().wait_for_event(&blas_event);
            }
            Step::TensorOp { args } => {
                args.run(&handles.cudnn);
            }
            Step::Gather {
                input,
                axis,
                indices,
                output,
            } => {
                assert!(
                    *axis == 1 && input.shape.rank() == 2,
                    "Gather only supported for rank 2 input and axis 1, got shape {:?} and axis {}",
                    input.shape,
                    axis
                );
                assert!(
                    indices.shape.rank() == 1 && indices.shape.has_simple_strides(),
                    "Gather indices must be rank-1 tensor with simple strides",
                );

                kernels::gather2dAxis1FloatFloat(
                    handles.cudnn.stream().inner(),
                    input.shape.shape()[0] as i32,
                    input.shape.shape()[1] as i32,
                    input.shape.strides()[0] as i32,
                    input.shape.strides()[1] as i32,
                    indices.shape.size() as i32,
                    input.ptr.ptr() as *const f32,
                    indices.ptr.ptr() as *const f32,
                    output.ptr.ptr() as *mut f32,
                )
                .unwrap();
            }
        }
    }
}

impl Debug for CudaExecutor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let indent = "    ";
        write!(
            f,
            "CudaExecutor {{\n    profile: {},\n    inputs: {:?},\n    outputs: {:?},\n    plan: {:?},\n}}",
            self.profile,
            debug_vec_multiline(indent, &self.inputs),
            debug_vec_multiline(indent, &self.outputs),
            debug_vec_multiline(indent, &self.steps),
        )
    }
}

impl Display for Profile {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Profile {{\n  steps: [\n")?;
        for step in &self.steps {
            writeln!(f, "    {}", step)?;
        }
        write!(f, "  ]\n\n")?;

        writeln!(f, "  Conv:      {:>10.4} ms", self.conv * 1e3)?;
        writeln!(f, "  Matmul:    {:>10.4} ms", self.mat_mul * 1e3)?;
        writeln!(f, "  Tensor op: {:>10.4} ms", self.tensor_op * 1e3)?;
        writeln!(f, "  Gather:    {:>10.4} ms", self.gather * 1e3)?;
        writeln!(f, "  ================")?;
        writeln!(f, "  Total GPU: {:>10.4} ms", self.total_gpu * 1e3)?;
        writeln!(f, "  Total CPU: {:>10.4} ms", self.total_cpu * 1e3)?;
        writeln!(f, "  Overhead:  {:>10.4} ms", self.timing_overhead * 1e3)?;

        writeln!(f, "}}")?;

        Ok(())
    }
}
