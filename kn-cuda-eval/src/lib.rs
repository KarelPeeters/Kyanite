#![warn(missing_debug_implementations)]

//! A Cuda CPU executor for neural network graphs from the `kn_graph` crate. The core type is [CudaExecutor](executor::CudaExecutor).
//!
//! The typical workflow:
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
//! let outputs: Vec<DTensor> = executor.evaluate_tensors(&inputs);
//! # Ok(())
//! # }
//! ```
//!
//! This crate is part of the [Kyanite](https://github.com/KarelPeeters/Kyanite) project, see its readme for more information.

/// Export the [Device] type for convenience: often an explicit dependency on the `kn_cuda_sys` crate is not needed.
pub use kn_cuda_sys::wrapper::handle::Device;

/// The autokernel infrastructure and specific kernels.
pub mod autokernel;
/// On-device tensor data structure.
pub mod device_tensor;
/// The main executor type and the compiler for it.
pub mod executor;
/// Quantization kernels. **Warning:** will be replaced autokernels.
pub mod quant;
/// A utility to automatically choose between CPU and GPU evaluation.
pub mod runtime;
/// Shape utilities.
pub mod shape;
/// Testing and debugging infrastructure.
pub mod tester;
/// Miscellaneous utilities.
pub mod util;

mod planner;

//TODO make this private again?
pub mod kernels;
mod offset_tensor;
mod step;
