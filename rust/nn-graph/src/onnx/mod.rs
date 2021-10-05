use std::path::Path;

use crate::graph::Graph;
use crate::onnx::load::load_onnx_impl;

mod proto;
//TODO switch back to include macro
// mod proto {
//     include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
// }

mod attributes;
mod store;
mod load;

pub fn load_onnx_graph(path: impl AsRef<Path>) -> Graph {
    load_onnx_impl(path.as_ref())
}