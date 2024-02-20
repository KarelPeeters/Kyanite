use std::path::Path;

use external_data::ExternalDataLoader;
pub use load::{InputShaper, OnnxDimValue};
pub use loader::GraphLoader;

use crate::graph::Graph;
use crate::onnx::result::OnnxResult;
use crate::shape::Size;

pub mod external_data;
mod inputs;
mod load;
#[allow(warnings)]
mod proto;
pub mod result;
mod store;
mod typed_value;
mod loader;

/// Load an [ONNX](https://github.com/onnx/onnx/blob/main/docs/IR.md) file from the given path.
///
/// If `allow_external` is true, the onnx will be allowed to load external data files,
/// see [the spec](https://github.com/onnx/onnx/blob/main/docs/IR.md#external-tensor-data).
/// If `allow_external` is false and the ONNX file does reference external data, an error is returned.
///
/// For more flexibility, see [GraphLoader].
pub fn load_graph_from_onnx_path(path: impl AsRef<Path>, allow_external: bool) -> OnnxResult<Graph> {
    let mut loader = GraphLoader::from_path(path, allow_external)?;
    loader.add_named_axis("batch_size", Size::BATCH);
    loader.load()
}

/// Load an [ONNX](https://github.com/onnx/onnx/blob/main/docs/IR.md) file from the given bytes.
///
/// The file is not allowed to reference external data files.
///
/// For more flexibility, see [GraphLoader].
pub fn load_graph_from_onnx_bytes(buffer: &[u8]) -> OnnxResult<Graph> {
    let mut loader = GraphLoader::from_bytes(buffer);
    loader.add_named_axis("batch_size", Size::BATCH);
    loader.load()
}

pub fn load_graph_from_onnx_bytes_custom(buffer: &[u8], external: Box<dyn ExternalDataLoader>) -> OnnxResult<Graph> {
    let mut loader = GraphLoader::from_bytes(buffer);
    loader.set_external_data(external);
    loader.add_named_axis("batch_size", Size::BATCH);
    loader.load()
}
