#![warn(missing_debug_implementations)]
#![allow(clippy::new_without_default)]

//! A neural network inference graph intermediate representation, with surrounding utilities.
//!
//! The core type of this crate is [Graph](graph::Graph),
//! see its documentation for how to manually build and compose graphs.
//!
//! An example demonstrating some of the features of this crate:
//! ```no_run
//! # use kn_graph::dot::graph_to_svg;
//! # use kn_graph::onnx::load_graph_from_onnx_path;
//! # use kn_graph::optimizer::optimize_graph;
//! # use kn_graph::cpu::{cpu_eval_graph, Tensor};
//! # use ndarray::{IxDyn, Array};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // load an onnx file
//! let graph = load_graph_from_onnx_path("test.onnx", false)?;
//! // optimize the graph
//! let graph = optimize_graph(&graph, Default::default());
//! // render the graph as an svg file
//! graph_to_svg("test.svg", &graph, false, false)?;
//! // execute the graph on the CPU
//! let batch_size = 8;
//! let inputs = [Tensor::zeros(IxDyn(&[batch_size, 16]))];
//! let outputs = cpu_eval_graph(&graph, batch_size, &inputs);
//! # Ok(())
//! # }
//! ```
//!
//! This crate is part of the [Kyanite](https://github.com/KarelPeeters/Kyanite) project, see its readme for more information.

// TODO write onnx load, optimize, run example

/// The [ndarray] crate is used for constant storage and CPU execution, and re-exported for convenience.
pub use ndarray;

/// The core graph datastructure.
pub mod graph;
/// Graph optimization.
pub mod optimizer;
/// The [Shape](shape::Shape) type and utilities.
pub mod shape;
/// The [DType](dtype::DType) enum.
pub mod dtype;

/// CPU graph execution.
pub mod cpu;
/// Graph visualization as a `dot` or `svg` file.
pub mod dot;
/// Onnx file loading.
pub mod onnx;
/// Hidden activations visualization.
pub mod visualize;

pub mod wrap_debug;
