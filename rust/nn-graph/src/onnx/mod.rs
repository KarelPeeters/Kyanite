use crate::graph::Graph;
use crate::onnx::external_data::{NoExternalData, PathExternalData};
use crate::onnx::load::graph_from_onnx_bytes;
use crate::onnx::result::OnnxResult;
use external_data::ExternalDataLoader;
use std::path::Path;

#[allow(warnings)]
mod proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

mod external_data;
mod inputs;
mod load;
pub mod result;
mod store;
mod typed_value;

pub fn load_graph_from_onnx_path(path: impl AsRef<Path>, allow_external: bool) -> OnnxResult<Graph> {
    let path = path.as_ref();
    let buf = std::fs::read(path).unwrap_or_else(|e| panic!("Failed to read input file {:?}, error {:?}", path, e));

    let external: Box<dyn ExternalDataLoader> = if allow_external {
        Box::new(PathExternalData(path.parent().unwrap().to_owned()))
    } else {
        Box::new(NoExternalData)
    };

    graph_from_onnx_bytes(&buf, &*external)
}

pub fn load_graph_from_onnx_bytes(buffer: &[u8]) -> OnnxResult<Graph> {
    graph_from_onnx_bytes(buffer, &NoExternalData)
}
