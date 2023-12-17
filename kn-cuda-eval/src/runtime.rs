use std::fmt::{Debug, Formatter};

use rand::{Rng, thread_rng};

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::cpu::cpu_eval_graph;
use kn_graph::dtype::DTensor;
use kn_graph::graph::Graph;

use crate::executor::CudaExecutor;

// TODO replace runtime with more lightweight "Device" struct that compiles single models at a time

/// A utility to dynamically choose between CPU and GPU evaluation at runtime.
pub struct Runtime {
    check: u64,
    core: RuntimeCore,
    infos: Vec<PreparedInfo>,
}

#[derive(Debug, Copy, Clone)]
pub struct PreparedToken {
    check: u64,
    index: usize,
}

#[derive(Debug)]
pub struct PreparedInfo {
    pub graph: Graph,
    pub batch_size: usize,
}

enum RuntimeCore {
    Cpu,
    Gpu {
        device: Device,
        executors: Vec<CudaExecutor>,
    },
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
            RuntimeCore::Cpu
        };
        Runtime {
            check,
            core,
            infos: vec![],
        }
    }

    pub fn prepare(&mut self, graph: Graph, batch_size: usize) -> PreparedToken {
        match &mut self.core {
            RuntimeCore::Cpu => {}
            RuntimeCore::Gpu { device, executors } => {
                assert_eq!(executors.len(), self.infos.len());
                executors.push(CudaExecutor::new(*device, &graph, batch_size));
            }
        }

        let info = PreparedInfo {
            graph: graph.clone(),
            batch_size,
        };

        let index = self.infos.len();
        self.infos.push(info);

        PreparedToken {
            check: self.check,
            index,
        }
    }

    // TODO move batch size from CPU into common struct
    pub fn info(&self, token: PreparedToken) -> &PreparedInfo {
        let PreparedToken { check, index } = token;
        assert_eq!(self.check, check);
        &self.infos[index]
    }

    pub fn eval(&mut self, token: PreparedToken, inputs: &[DTensor]) -> Vec<DTensor> {
        let PreparedToken { check, index } = token;
        assert_eq!(self.check, check);

        match &mut self.core {
            RuntimeCore::Cpu => {
                let info = &self.infos[index];
                cpu_eval_graph(&info.graph, info.batch_size, inputs)
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
            RuntimeCore::Cpu => None,
            RuntimeCore::Gpu { device, executors: _ } => Some(*device),
        };

        f.debug_struct("Runtime")
            .field("check", &self.check)
            .field("device", &device)
            .finish()
    }
}
