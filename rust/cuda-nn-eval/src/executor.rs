use std::cmp::max;
use std::fmt::{Debug, Formatter};
use std::time::Instant;

use bytemuck::{cast_slice, cast_slice_mut};

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

    stage: Vec<f32>,
    outputs: Vec<Vec<f32>>,

    profile: bool,
}

#[derive(Debug)]
pub(crate) struct Handles {
    pub cudnn: CudnnHandle,
    pub cublas: CublasHandle,
}

#[derive(Debug)]
pub enum Step {
    CopyInput { index: usize, mem: DeviceMem },
    Conv { args: FusedConvolutionArgs },
    MatMul { args: BatchedMatMulArgs },
    TensorOp { args: TensorOpArgs },
    Gather { input: Tensor, axis: usize, indices: Tensor, output: Tensor },
    CopyOutput { index: usize, tensor: Tensor },
}

impl CudnnExecutor {
    pub fn new(device: Device, graph: &Graph, batch_size: usize) -> Self {
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

        CudnnExecutor { handles, plan, stage, outputs, profile: false }
    }

    pub fn evaluate(&mut self, inputs: &[&[f32]]) -> &[Vec<f32>] {
        let mut timers = vec![];
        let start_cpu = Instant::now();
        let start_all;
        let end_all;

        unsafe {
            start_all = self.handles.cudnn.stream().record_new_event();

            for step in &self.plan {
                let start = self.handles.cudnn.stream().record_new_event();
                step.run(&self.handles, inputs, &mut self.stage, &mut self.outputs);
                let end = self.handles.cudnn.stream().record_new_event();

                timers.push((step, start, end));
            }

            end_all = self.handles.cudnn.stream().record_new_event();
        }

        let end_cpu = Instant::now();

        if self.profile {
            let mut conv_time = 0.0;
            let mut mat_mul_time = 0.0;
            let mut tensor_op_time = 0.0;
            let mut gather_time = 0.0;
            let mut copy_to_device_time = 0.0;
            let mut copy_to_host_time = 0.0;

            for (i, (step, start, end)) in timers.iter().enumerate() {
                let time = end.time_elapsed_since(start);

                *match step {
                    Step::CopyInput { .. } => &mut copy_to_device_time,
                    Step::Conv { .. } => &mut conv_time,
                    Step::MatMul { .. } => &mut mat_mul_time,
                    Step::TensorOp { .. } => &mut tensor_op_time,
                    Step::Gather { .. } => &mut gather_time,
                    Step::CopyOutput { .. } => &mut copy_to_host_time,
                } += time;

                println!("{: >4} time {:.4} ms, step {:?}", i, time, step);
            }

            println!("Conv:      {:.4} ms", conv_time);
            println!("Matmul:    {:.4} ms", mat_mul_time);
            println!("Tensor op: {:.4} ms", tensor_op_time);
            println!("Gather:    {:.4} ms", gather_time);
            println!("Copy ->:   {:.4} ms", copy_to_device_time);
            println!("Copy <-:   {:.4} ms", copy_to_host_time);
            println!("================");
            println!("Total GPU: {:.4} ms", end_all.time_elapsed_since(&start_all));
            println!("Total CPU: {:.4} ms", (end_cpu - start_cpu).as_secs_f32() * 1000.0);
        }

        &self.outputs
    }

    pub fn set_profile(&mut self, profile: bool) {
        self.profile = profile;
    }
}

impl Step {
    unsafe fn run(&self, handles: &Handles, inputs: &[&[f32]], stage: &mut [f32], outputs: &mut [Vec<f32>]) {
        match self {
            Step::CopyInput { index, mem } => {
                mem.copy_from_host(cast_slice(inputs[*index]))
            }
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
            Step::Gather { input, axis, indices, output } => {
                assert!(
                    *axis == 1 && input.shape.rank() == 2,
                    "Gather only supported for rank 2 input and axis 1, got shape {:?} and axis {}",
                    input.shape, axis
                );
                assert!(
                    indices.shape.rank() == 1 && indices.shape.has_simple_strides(),
                    "Gather indices must be rank-1 tensor with simple strides",
                );

                kernels::gather2dAxis1FloatFloat(
                    handles.cudnn.stream().inner(),
                    input.shape.shape()[0] as i32, input.shape.shape()[1] as i32,
                    input.shape.strides()[0] as i32, input.shape.strides()[1] as i32,
                    indices.shape.size() as i32,
                    input.mem.ptr() as *const f32, indices.mem.ptr() as *const f32, output.mem.ptr() as *mut f32,
                ).unwrap();
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
        write!(f, "CudnnExecutor {{\n    profile: {},\n    handle: {:?},\n    plan: [\n", self.profile, self.handles)?;

        for step in &self.plan {
            writeln!(f, "        {:?},", step)?;
        }

        writeln!(f, "    ]\n}}")?;
        Ok(())
    }
}