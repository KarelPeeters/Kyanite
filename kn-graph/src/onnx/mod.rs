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

pub mod external_data;
mod inputs;
mod load;
pub mod result;
mod store;
pub(crate) mod typed_value;

/// Load an [ONNX](https://github.com/onnx/onnx/blob/main/docs/IR.md) file from the given path.
///
/// If `allow_external` is true, the onnx will be allowed to load external data files,
/// see [the spec](https://github.com/onnx/onnx/blob/main/docs/IR.md#external-tensor-data).
/// If `allow_external` is false and the ONNX file does reference external data, an error is returned.
pub fn load_graph_from_onnx_path(path: impl AsRef<Path>, allow_external: bool) -> OnnxResult<Graph> {
    let path = path.as_ref();
    let buf = std::fs::read(path).to_onnx_result(path)?;

    let mut external: Box<dyn ExternalDataLoader> = if allow_external {
        let parent = path
            .parent()
            .ok_or_else(|| OnnxError::MustHaveParentPath(path.to_owned()))?;
        Box::new(PathExternalData(parent.to_owned()))
    } else {
        Box::new(NoExternalData)
    };

    graph_from_onnx_bytes(&buf, &mut *external)
}

/// Load an [ONNX](https://github.com/onnx/onnx/blob/main/docs/IR.md) file from the given bytes.
///
/// The file is not allowed to reference external data files.
pub fn load_graph_from_onnx_bytes(buffer: &[u8]) -> OnnxResult<Graph> {
    graph_from_onnx_bytes(buffer, &mut NoExternalData)
}

pub fn load_graph_from_onnx_bytes_custom(buffer: &[u8], external: &mut dyn ExternalDataLoader) -> OnnxResult<Graph> {
    graph_from_onnx_bytes(buffer, external)
}