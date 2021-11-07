use std::cmp::max;
use std::fmt::{Debug, Formatter};

use bytemuck::{cast_slice, cast_slice_mut};

use cuda_sys::wrapper::event::CudaEvent;
use cuda_sys::wrapper::handle::{CudnnHandle, Device};
use nn_graph::graph::{Graph, ValueInfo};

use crate::planner::{Planner, Step};

pub struct CudnnExecutor {
    handle: CudnnHandle,
    plan: Vec<Step>,

    stage: Vec<f32>,
    outputs: Vec<Vec<f32>>,

    profile: bool,
}

impl CudnnExecutor {
    pub fn new(device: Device, graph: &Graph, batch_size: usize) -> Self {
        let handle = CudnnHandle::new(device);
        let mut planner = Planner::new(handle);

        for value in graph.values() {
            let ValueInfo { shape, operation } = &graph[value];
            planner.visit(value, shape.eval(batch_size), operation);
        }

        let mut outputs = vec![];
        let mut stage_size = 0;

        for (index, &value) in graph.outputs().iter().enumerate() {
            let tensor = planner.visit_output(index, value);

            if !tensor.shape.has_simple_strides() {
                stage_size = max(stage_size, tensor.mem.len() / 4);
            }

            outputs.push(vec![f32::NAN; tensor.shape.size()])
        }

        let (handle, plan) = planner.finish();

        let stage = vec![f32::NAN; stage_size];
        CudnnExecutor { handle, plan, stage, outputs, profile: false }
    }

    pub fn evaluate(&mut self, inputs: &[&[f32]]) -> &[Vec<f32>] {
        let mut timers = vec![];

        unsafe {
            for step in &self.plan {
                let start = CudaEvent::new();
                let end = CudaEvent::new();

                self.handle.stream().record_event(&start);
                step.run(&mut self.handle, inputs, &mut self.stage, &mut self.outputs);
                self.handle.stream().record_event(&end);

                timers.push((step, start, end));
            }
        }

        if self.profile {
            let mut conv_time = 0.0;
            let mut tensor_op_time = 0.0;
            let mut copy_to_device_time = 0.0;
            let mut copy_to_host_time = 0.0;

            for (i, (step, start, end)) in timers.iter().enumerate() {
                let time = end.time_elapsed_since(start);

                *match step {
                    Step::CopyInput { .. } => &mut copy_to_device_time,
                    Step::Conv { .. } => &mut conv_time,
                    Step::TensorOp { .. } => &mut tensor_op_time,
                    Step::CopyOutput { .. } => &mut copy_to_host_time,
                } += time;

                println!("{: >4} time {:.4} ms, step {:?}", i, time, step);
            }

            println!("Conv:      {:.4}", conv_time);
            println!("Tensor op: {:.4}", tensor_op_time);
            println!("Copy ->:   {:.4}", copy_to_device_time);
            println!("Copy <-:   {:.4}", copy_to_host_time);
        }

        &self.outputs
    }

    pub fn set_profile(&mut self, profile: bool) {
        self.profile = profile;
    }
}

impl Step {
    unsafe fn run(&self, handle: &mut CudnnHandle, inputs: &[&[f32]], stage: &mut [f32], outputs: &mut [Vec<f32>]) {
        match self {
            Step::CopyInput { index, mem } => {
                mem.copy_from_host(cast_slice(inputs[*index]))
            }
            Step::Conv { details: _, args } => {
                args.run(handle);
            }
            Step::TensorOp { args } => {
                args.run(handle);
            }
            //TODO look into fusing the copy operation if multiple outputs are sliced views on the same value
            Step::CopyOutput { index, tensor } => {
                let index = *index;

                if tensor.shape.has_simple_strides() {
                    // directly copy everything into the output
                    tensor.mem.copy_to_host(cast_slice_mut(&mut outputs[index]));
                } else {
                    // copy the entire mem over
                    let used_stage = &mut stage[0..tensor.mem.len() / 4];
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
        f.debug_struct("CudnnExecutor")
            .field("plan", &self.plan)
            .field("profile", &self.profile)
            .field("handle", &self.handle)
            .finish()
    }
}