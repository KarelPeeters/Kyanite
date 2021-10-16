use std::cmp::max;

use bytemuck::{cast_slice, cast_slice_mut};

use cuda_sys::wrapper::handle::{CudnnHandle, Device};
use nn_graph::graph::{Graph, ValueInfo};

use crate::planner::{Planner, Step};

#[derive(Debug)]
pub struct CudnnExecutor {
    handle: CudnnHandle,
    plan: Vec<Step>,

    stage: Vec<f32>,
    outputs: Vec<Vec<f32>>,
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
        CudnnExecutor { handle, plan, stage, outputs }
    }

    pub fn evaluate(&mut self, inputs: &[&[f32]]) -> &[Vec<f32>] {
        unsafe {
            for step in &self.plan {
                match step {
                    Step::CopyInput { index, mem } => {
                        mem.copy_from_host(cast_slice(inputs[*index]))
                    }
                    Step::Conv { args } => {
                        args.run(&mut self.handle);
                    }
                    Step::TensorOp { args } => {
                        args.run(&mut self.handle);
                    }
                    Step::CopyOutput { index, tensor } => {
                        //TODO look into fusing the copy operation if multiple outputs are sliced views on the same value

                        if tensor.shape.has_simple_strides() {
                            // directly copy everything into the output
                            tensor.mem.copy_to_host(cast_slice_mut(&mut self.outputs[*index]));
                        } else {
                            // copy the entire mem over
                            let stage = &mut self.stage[0..tensor.mem.len() / 4];
                            tensor.mem.copy_to_host(cast_slice_mut(stage));

                            // selectively copy over the actual values we want
                            let output = &mut self.outputs[*index];
                            let mut output_i = 0;
                            tensor.shape.visit_strided_indices(|stage_i| {
                                output[output_i] = stage[stage_i];
                                output_i += 1;
                            });
                        }
                    }
                }
            }
        }

        &self.outputs
    }
}