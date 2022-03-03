use std::cmp::max;
use std::fmt::{Debug, Display, Formatter};
use std::time::Instant;

use bytemuck::{cast_slice, cast_slice_mut};
use itertools::Itertools;

use cuda_sys::wrapper::graph::CudaGraphExec;
use cuda_sys::wrapper::group::{BatchedMatMulArgs, FusedConvolutionArgs, TensorOpArgs};
use cuda_sys::wrapper::handle::{CublasHandle, CudnnHandle, Device};
use cuda_sys::wrapper::mem::device::DeviceMem;
use cuda_sys::wrapper::status::Status;
use nn_graph::graph::Graph;

use crate::kernels;
use crate::planner::Planner;
use crate::tensor::Tensor;

pub struct CudnnExecutor {
    handles: Handles,
    plan: Vec<Step>,
    graph_plan: Option<GraphPlan>,

    stage: Vec<f32>,
    outputs: Vec<Vec<f32>>,

    use_graph: bool,
    profile: bool,
    last_profile: Option<Profile>,
}

struct GraphPlan {
    steps_before: Vec<usize>,
    graph_exec: CudaGraphExec,
    steps_after: Vec<usize>,
}

#[derive(Debug)]
pub(crate) struct Handles {
    pub cudnn: CudnnHandle,
    pub cublas: CublasHandle,
}

#[derive(Debug)]
pub enum Step {
    CopyInput {
        index: usize,
        mem: DeviceMem,
    },
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
        input: Tensor,
        axis: usize,
        indices: Tensor,
        output: Tensor,
    },
    CopyOutput {
        index: usize,
        tensor: Tensor,
    },
}

#[derive(Default, Debug, Clone)]
pub struct Profile {
    pub steps: Vec<String>,

    pub conv: f32,
    pub mat_mul: f32,
    pub tensor_op: f32,
    pub gather: f32,
    pub copy_to_device: f32,
    pub copy_to_host: f32,

    pub total_cpu: f32,
    pub total_gpu: f32,
    pub timing_overhead: f32,
}

impl CudnnExecutor {
    pub fn new(device: Device, graph: &Graph, batch_size: usize, use_graph: bool) -> Self {
        let handles = Handles {
            cudnn: CudnnHandle::new(device),
            cublas: CublasHandle::new(device),
        };
        let mut planner = Planner::new(&handles, graph, batch_size);

        // do all necessary calculations
        for &output in graph.outputs() {
            planner.visit(output);
        }

        // schedule copy operations
        let mut outputs = vec![];
        let mut stage_size = 0;

        for (index, &value) in graph.outputs().iter().enumerate() {
            let tensor = planner.copy_output(index, value);

            if !tensor.shape.has_simple_strides() {
                stage_size = max(stage_size, tensor.mem.len_bytes() / 4);
            }

            outputs.push(vec![f32::NAN; tensor.shape.size()])
        }

        let stage = vec![f32::NAN; stage_size];
        let plan = planner.finish();

        let graph_plan = if use_graph {
            Some(record_graph(&handles, &plan))
        } else {
            None
        };

        CudnnExecutor {
            handles,
            plan,
            graph_plan,
            stage,
            outputs,
            use_graph: true,
            profile: false,
            last_profile: None,
        }
    }

    pub fn evaluate(&mut self, inputs: &[&[f32]]) -> &[Vec<f32>] {
        let mut timers = vec![];
        let start_cpu = Instant::now();
        let start_all;
        let end_all;

        unsafe {
            start_all = self.handles.cudnn.stream().record_new_event();

            if let Some(graph_plan) = &self.graph_plan {
                for &step in &graph_plan.steps_before {
                    self.plan[step].run(&self.handles, inputs, &mut self.stage, &mut self.outputs);
                }

                graph_plan.graph_exec.launch(self.handles.cudnn.stream());

                for &step in &graph_plan.steps_after {
                    self.plan[step].run(&self.handles, inputs, &mut self.stage, &mut self.outputs);
                }
            } else {
                for step in &self.plan {
                    let start = self.handles.cudnn.stream().record_new_event();
                    step.run(&self.handles, inputs, &mut self.stage, &mut self.outputs);
                    let end = self.handles.cudnn.stream().record_new_event();

                    if self.profile {
                        timers.push((step, start, end));
                    }
                }
            }

            end_all = self.handles.cudnn.stream().record_new_event();
            self.handles.cudnn.stream().synchronize();
        }

        let end_cpu = Instant::now();

        if self.profile {
            let mut profile = Profile::default();

            for (i, (step, start, end)) in timers.iter().enumerate() {
                let time = end.time_elapsed_since(start);

                *match step {
                    Step::CopyInput { .. } => &mut profile.copy_to_device,
                    Step::Conv { .. } => &mut profile.conv,
                    Step::MatMul { .. } => &mut profile.mat_mul,
                    Step::TensorOp { .. } => &mut profile.tensor_op,
                    Step::Gather { .. } => &mut profile.gather,
                    Step::CopyOutput { .. } => &mut profile.copy_to_host,
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
        } else {
            self.last_profile = None;
        }

        &self.outputs
    }

    pub fn use_graph(&mut self, use_graph: bool) {
        self.use_graph = use_graph;
    }

    pub fn set_profile(&mut self, profile: bool) {
        self.profile = profile;
    }

    pub fn last_profile(&self) -> Option<&Profile> {
        self.last_profile.as_ref()
    }
}

fn record_graph(handles: &Handles, steps: &[Step]) -> GraphPlan {
    let mut steps_before = vec![];
    let mut steps_after = vec![];

    unsafe {
        handles.cudnn.stream().begin_capture();

        for (i, step) in steps.iter().enumerate() {
            match step {
                Step::CopyInput { .. } => {
                    steps_before.push(i);
                    continue;
                }
                Step::CopyOutput { .. } => {
                    steps_after.push(i);
                    continue;
                }
                Step::Conv { .. } | Step::MatMul { .. } | Step::TensorOp { .. } | Step::Gather { .. } => {
                    step.run(handles, &[], &mut [], &mut []);
                }
            }
        }

        assert_eq!(steps_before, (0..steps_before.len()).collect_vec());
        assert_eq!(
            steps_after,
            ((steps.len() - steps_after.len())..steps.len()).collect_vec()
        );

        let graph = handles.cudnn.stream().end_capture();
        let graph_exec = graph.instantiate();

        GraphPlan {
            steps_before,
            graph_exec,
            steps_after,
        }
    }
}

impl Step {
    unsafe fn run(&self, handles: &Handles, inputs: &[&[f32]], stage: &mut [f32], outputs: &mut [Vec<f32>]) {
        match self {
            Step::CopyInput { index, mem } => mem.copy_from_host(cast_slice(inputs[*index])),
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
                    input.mem.ptr() as *const f32,
                    indices.mem.ptr() as *const f32,
                    output.mem.ptr() as *mut f32,
                )
                .unwrap();
            }
            //TODO look into fusing the copy operation if multiple outputs are sliced views on the same value
            //  this has recently become easier now that restriding is available
            Step::CopyOutput { index, tensor } => {
                let index = *index;

                if tensor.shape.has_simple_strides() {
                    // directly copy everything into the output
                    tensor.mem.copy_to_host(cast_slice_mut(&mut outputs[index]));
                } else {
                    // copy the entire mem over
                    let used_stage = &mut stage[0..tensor.mem.len_bytes() / 4];
                    tensor.mem.copy_to_host(cast_slice_mut(used_stage));

                    // selectively copy over the actual values we want
                    let output = &mut outputs[index];
                    let mut output_i = 0;
                    tensor.shape.visit_strided_indices(|stage_i| {
                        output[output_i] = used_stage[stage_i];
                        output_i += 1;
                    });
                }
            }
        }
    }
}

impl Debug for CudnnExecutor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CudnnExecutor {{\n    profile: {},\n    handle: {:?},\n    use_graph: {}, plan: [\n",
            self.profile, self.handles, self.use_graph
        )?;

        for step in &self.plan {
            writeln!(f, "        {:?},", step)?;
        }

        writeln!(f, "    ]\n}}")?;
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

        writeln!(f, "  Conv:      {:>10.4} ms", self.conv * 1e3)?;
        writeln!(f, "  Matmul:    {:>10.4} ms", self.mat_mul * 1e3)?;
        writeln!(f, "  Tensor op: {:>10.4} ms", self.tensor_op * 1e3)?;
        writeln!(f, "  Gather:    {:>10.4} ms", self.gather * 1e3)?;
        writeln!(f, "  Copy ->:   {:>10.4} ms", self.copy_to_device * 1e3)?;
        writeln!(f, "  Copy <-:   {:>10.4} ms", self.copy_to_host * 1e3)?;
        writeln!(f, "  ================")?;
        writeln!(f, "  Total GPU: {:>10.4} ms", self.total_gpu * 1e3)?;
        writeln!(f, "  Total CPU: {:>10.4} ms", self.total_cpu * 1e3)?;
        writeln!(f, "  Overhead:  {:>10.4} ms", self.timing_overhead * 1e3)?;

        writeln!(f, "}}")?;

        Ok(())
    }
}
