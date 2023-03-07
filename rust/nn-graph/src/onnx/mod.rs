use std::path::Path;

use external_data::ExternalDataLoader;

use crate::graph::Graph;
use crate::onnx::external_data::{NoExternalData, PathExternalData};
use crate::onnx::load::graph_from_onnx_bytes;
use crate::onnx::result::{OnnxError, OnnxResult, ToOnnxLoadResult};

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
    let buf = std::fs::read(path).to_onnx_result(path)?;

    let external: Box<dyn ExternalDataLoader> = if allow_external {
        let parent = path
            .parent()
            .ok_or_else(|| OnnxError::MustHaveParentPath(path.to_owned()))?;
        Box::new(PathExternalData(parent.to_owned()))
    } else {
        Box::new(NoExternalData)
    };

    graph_from_onnx_bytes(&buf, &*external)
}

pub fn load_graph_from_onnx_bytes(buffer: &[u8]) -> OnnxResult<Graph> {
    graph_from_onnx_bytes(buffer, &NoExternalData)
}
