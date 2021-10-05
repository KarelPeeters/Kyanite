use bytemuck::{cast_slice, cast_slice_mut};
use itertools::Itertools;

use cuda_sys::wrapper::handle::CudnnHandle;
use nn_graph::graph::{Graph, ValueInfo};

use crate::planner::{Planner, Step};

#[derive(Debug)]
pub struct CudnnExecutor {
    handle: CudnnHandle,
    plan: Vec<Step>,
    outputs: Vec<Vec<f32>>,
}

impl CudnnExecutor {
    pub fn new(handle: CudnnHandle, graph: &Graph, batch_size: usize) -> Self {
        let mut planner = Planner::new(handle);

        for value in graph.values() {
            let ValueInfo { shape, operation } = &graph[value];
            planner.visit(value, &shape.eval(batch_size), operation);
        }

        for (index, &value) in graph.outputs().iter().enumerate() {
            planner.visit_output(index, value);
        }

        let (handle, plan) = planner.finish();

        let outputs = graph.outputs().iter().map(|&output| {
            let size = graph[output].shape.size().eval(batch_size);
            vec![f32::NAN; size]
        }).collect_vec();

        CudnnExecutor { handle, plan, outputs }
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
                    Step::CopyOutput { index, mem } => {
                        //TODO properly handle strided outputs here
                        if mem.len_bytes() == self.outputs[*index].len() * 4 {
                            mem.copy_to_host(cast_slice_mut(&mut self.outputs[*index]))
                        }
                    }
                }
            }
        }

        &self.outputs
    }
}