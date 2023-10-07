#![warn(missing_debug_implementations)]

//! A Cuda CPU executor for neural network graphs from the `kn_graph` crate. The core type is [CudaExecutor](executor::CudaExecutor).
//!
//! This crate is part of the [Kyanite](https://github.com/KarelPeeters/Kyanite) project, see its readme for more information.
//! See [system-requirements](https://github.com/KarelPeeters/Kyanite#system-requirements) for how to set up the cuda libraries.
//!
//! # Quick demo
//!
//! ```no_run
//! # use kn_cuda_eval::executor::CudaExecutor;
//! # use kn_cuda_sys::wrapper::handle::Device;
//! # use kn_graph::dtype::{DTensor, Tensor};
//! # use kn_graph::ndarray::{Array, IxDyn};
//! # use kn_graph::onnx::load_graph_from_onnx_path;
//! # use kn_graph::optimizer::optimize_graph;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // load and optimize the graph
//! let graph = load_graph_from_onnx_path("test.onnx", false)?;
//! let graph = optimize_graph(&graph, Default::default());
//!
//! // select a device
//! let device = Device::new(0);
//!
//! // build an executor
//! let batch_size = 8;
//! let mut executor = CudaExecutor::new(device, &graph, batch_size);
//!
//! // evaluate the graph with some inputs, get the outputs back
//! let inputs = [DTensor::F32(Tensor::zeros(vec![batch_size, 16]))];
//! let outputs: &[DTensor] = executor.evaluate(&inputs);
//! # Ok(())
//! # }
//! ```


/// Export the [Device] type for convenience: often an explicit dependency on the `kn_cuda_sys` crate is not needed.
pub use kn_cuda_sys::wrapper::handle::Device;

/// The autokernel infrastructure and specific kernels.
pub mod autokernel;
/// On-device tensor data structure.
pub mod device_tensor;
/// The main executor type and the compiler for it.
pub mod executor;
/// A utility to automatically choose between CPU and GPU evaluation.
pub mod runtime;
/// Shape utilities.
pub mod shape;
/// Testing and debugging infrastructure.
pub mod tester;
/// Miscellaneous utilities.
pub mod util;
/// Tensor utility.
pub mod offset_tensor;

mod planner;
mod step;
