#![warn(missing_debug_implementations)]

//! A wrapper crate around `kn-graph` and `kn-cuda-eval` that allows selecting whether to use a CPU or GPU at runtime
//! through the [Device] type.
//!
//! By default this crate only includes the [Device::Cpu] device, and only depends on `kn-graph`.
//! To enable the [Device::Cuda] device, enable the `cuda` cargo feature.
//! This adds a dependency on `kn-cuda-eval` and the cuda libraries.
//!
//! This crate is part of the [Kyanite](https://github.com/KarelPeeters/Kyanite) project, see its readme for more information.
//! See [system-requirements](https://github.com/KarelPeeters/Kyanite#system-requirements) for how to set up the cuda libraries.
//!
//! # Quick demo
//!
//! ```no_run
//! # use kn_cuda_sys::wrapper::handle::CudaDevice;
//! # use kn_graph::dtype::{DTensor, Tensor};
//! # use kn_graph::onnx::load_graph_from_onnx_path;
//! # use kn_graph::optimizer::optimize_graph;
//! # use kn_runtime::Device;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // load and optimize a graph
//! let graph = load_graph_from_onnx_path("test.onnx", false)?;
//! let graph = optimize_graph(&graph, Default::default());
//!
//! // select a device, at runtime
//! let device_str = "cpu";
//! let device = match device_str {
//!     "auto" => Device::best(),
//!     "cpu" => Device::Cpu,
//!     "cuda" => Device::Cuda(CudaDevice::all().next().unwrap()),
//!     _ => panic!("unknown device"),
//! };
//!
//! // prepare the graph for execution
//! let batch_size = 8;
//! let prepared = device.prepare(graph, batch_size);
//!
//! // evaluate the graph with some inputs, get the outputs back
//! let inputs = [DTensor::F32(Tensor::zeros(vec![batch_size, 16]))];
//! let outputs: &[DTensor] = prepared.evaluate(&inputs);
//! # Ok(())
//! # }
//! ```

use std::fmt::Debug;

#[cfg(feature = "cuda")]
pub use kn_cuda_eval::executor::CudaExecutor;
#[cfg(feature = "cuda")]
pub use kn_cuda_sys::wrapper::handle::CudaDevice;
use kn_graph::cpu::cpu_eval_graph;
use kn_graph::dtype::DTensor;
use kn_graph::graph::Graph;

/// Whether the crate was compiled with cuda support.
///
/// This is independent of whether the current system actually has a cuda device available.
pub fn compiled_with_cuda_support() -> bool {
    #[cfg(feature = "cuda")]
    return true;
    #[cfg(not(feature = "cuda"))]
    return false;
}

/// A device that can be used to evaluate a graph.
#[derive(Debug)]
pub enum Device {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(CudaDevice),
}

/// A graph that has been prepared for evaluation on a device.
/// * For a CPU this is just the graph itself and a hardcoded batch size.
/// * For a GPU this is a fully planned and memory allocated execution plan.
#[derive(Debug)]
pub enum PreparedGraph {
    CPU { graph: Graph, batch_size: usize },
    #[cfg(feature = "cuda")]
    Cuda { executor: CudaExecutor },
}

impl Device {
    pub fn prepare(&self, graph: Graph, batch_size: usize) -> PreparedGraph {
        match *self {
            Device::Cpu => PreparedGraph::CPU { graph, batch_size },
            #[cfg(feature = "cuda")]
            Device::Cuda(device) => PreparedGraph::Cuda {
                executor: CudaExecutor::new(device, &graph, batch_size),
            },
        }
    }

    /// Returns the best available device.
    ///
    /// For now the algorithm used is very simple:
    /// * pick the first cuda device if any are available
    /// * otherwise use the CPU
    pub fn best() -> Device {
        if let Some(device) = Device::first_cuda() {
            return device;
        }

        Device::Cpu
    }

    /// Returns the first available cuda device if any.
    pub fn first_cuda() -> Option<Device> {
        #[cfg(feature = "cuda")]
        if let Some(device) = CudaDevice::all().next()? {
            return Some(Device::Cuda(device));
        }

        None
    }
}

impl PreparedGraph {
    pub fn eval(&mut self, inputs: &[DTensor]) -> Vec<DTensor> {
        match self {
            PreparedGraph::CPU { graph, batch_size } => cpu_eval_graph(graph, *batch_size, inputs),
            #[cfg(feature = "cuda")]
            PreparedGraph::Cuda { executor } => executor.evaluate(inputs).to_owned(),
        }
    }
}
