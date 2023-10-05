use std::fmt::{Debug, Formatter};

use rand::{Rng, thread_rng};

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::cpu::cpu_eval_graph;
use kn_graph::dtype::DTensor;
use kn_graph::graph::Graph;

use crate::executor::CudaExecutor;

/// A utility to dynamically choose between CPU and GPU evaluation at runtime.
pub struct Runtime {
    check: u64,
    core: RuntimeCore,
}

enum RuntimeCore {
    Cpu(Vec<(Graph, usize)>),
    Gpu {
        device: Device,
        executors: Vec<CudaExecutor>,
    },
}

#[derive(Debug, Copy, Clone)]
pub struct GraphToken {
    check: u64,
    index: usize,
}

impl Runtime {
    pub fn new(gpu_device: Option<Device>) -> Self {
        let check = thread_rng().gen();
        let core = if let Some(device) = gpu_device {
            RuntimeCore::Gpu {
                device,
                executors: vec![],
            }
        } else {
            RuntimeCore::Cpu(vec![])
        };
        Runtime { check, core }
    }

    pub fn prepare(&mut self, graph: Graph, batch_size: usize) -> GraphToken {
        let index = match &mut self.core {
            RuntimeCore::Cpu(graphs) => {
                let index = graphs.len();
                graphs.push((graph, batch_size));
                index
            }
            RuntimeCore::Gpu { device, executors } => {
                let index = executors.len();
                executors.push(CudaExecutor::new(*device, &graph, batch_size));
                index
            }
        };

        GraphToken {
            check: self.check,
            index,
        }
    }

    pub fn eval(&mut self, token: GraphToken, inputs: &[DTensor]) -> Vec<DTensor> {
        let GraphToken { check, index } = token;
        assert_eq!(self.check, check);

        match &mut self.core {
            RuntimeCore::Cpu(graphs) => {
                let (graph, batch_size) = &graphs[index];
                cpu_eval_graph(graph, *batch_size, inputs)
            }
            RuntimeCore::Gpu { device: _, executors } => {
                let executor = &mut executors[index];
                executor.evaluate(inputs).to_owned()
            }
        }
    }
}

impl Debug for Runtime {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let device = match &self.core {
            RuntimeCore::Cpu(_) => None,
            RuntimeCore::Gpu { device, executors: _ } => Some(*device),
        };

        f.debug_struct("Runtime")
            .field("check", &self.check)
            .field("device", &device)
            .finish()
    }
}
