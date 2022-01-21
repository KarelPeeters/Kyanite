use std::path::Path;

use crate::graph::Graph;
use crate::onnx::load::{load_model_proto, onnx_proto_to_graph};

#[allow(warnings)]
mod proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

mod attributes;
mod store;
mod load;
mod typed_value;

pub fn load_graph_from_onnx_path(path: impl AsRef<Path>) -> Graph {
    let path = path.as_ref();
    let buf = std::fs::read(path)
        .unwrap_or_else(|e| panic!("Failed to read input file {:?}, error {:?}", path, e));
    load_graph_from_onnx_bytes(&buf)
}

pub fn load_graph_from_onnx_bytes(buf: &[u8]) -> Graph {
    let model = load_model_proto(buf);
    onnx_proto_to_graph(&model)
}