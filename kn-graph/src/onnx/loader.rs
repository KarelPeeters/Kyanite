use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Deref;
use std::path::Path;

use crate::graph::Graph;
use crate::onnx::{InputShaper, OnnxDimValue};
use crate::onnx::external_data::{ExternalDataLoader, NoExternalData, PathExternalData};
use crate::onnx::load::graph_from_onnx_bytes;
use crate::onnx::result::{OnnxError, OnnxResult, ToOnnxLoadResult};
use crate::shape::{Shape, Size};

/// Load an [ONNX](https://github.com/onnx/onnx/blob/main/docs/IR.md) graph.
///
/// Many loading settings are customizable:
/// * the source, either from a path through [Self::from_path] or from bytes through [Self::from_bytes].
/// * whether [external data](https://github.com/onnx/onnx/blob/main/docs/ExternalData.md) is allowed,
///     through [Self::from_path] `allow_external` or [Self::set_external_data].
/// * input shape overrides (in order of priority):
///   * fully custom through [Self::set_input_shaper_custom]
///   * specific input overrides through [Self::force_input_shapes]
///   * named axes through [Self::add_named_axis]
///
/// A simple example:
/// ```no_run
/// # use kn_graph::graph::Graph;
/// # use kn_graph::onnx::GraphLoader;
/// # use kn_graph::shape;
/// # use kn_graph::shape::Size;
/// // load from a path, disallowing external data
/// let mut loader = GraphLoader::from_path("model.onnx", false).unwrap();
/// // set some named axes
/// loader.add_named_axis("batch_size", Size::BATCH);
/// loader.add_named_axis("sequence_length", Size::fixed(128));
/// // override the third input shape
/// loader.force_input_shapes(vec![None, None, Some(shape![1, Size::BATCH, 3])]);
/// // load the graph
/// let graph = loader.load().unwrap();
/// ```
#[allow(missing_debug_implementations)]
pub struct GraphLoader<'a> {
    bytes: Cow<'a, [u8]>,
    external: Box<dyn ExternalDataLoader>,

    // input shape overrides
    input_shaper_custom: Option<Box<InputShaper>>,
    input_shape_overrides: Option<Vec<Option<Shape>>>,
    named_axes: HashMap<String, Size>,
}

impl<'a> GraphLoader<'a> {
    pub fn from_path(path: impl AsRef<Path>, allow_external: bool) -> OnnxResult<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).to_onnx_result(path)?;

        let external: Box<dyn ExternalDataLoader> = if allow_external {
            let parent = path
                .parent()
                .ok_or_else(|| OnnxError::MustHaveParentPath(path.to_owned()))?;
            Box::new(PathExternalData(parent.to_owned()))
        } else {
            Box::new(NoExternalData)
        };

        Ok(GraphLoader {
            bytes: Cow::Owned(bytes),
            external,

            input_shaper_custom: None,
            input_shape_overrides: None,
            named_axes: HashMap::new(),
        })
    }

    pub fn from_bytes(bytes: &'a [u8]) -> Self {
        GraphLoader {
            bytes: Cow::Borrowed(bytes),
            external: Box::new(NoExternalData),

            input_shaper_custom: None,
            input_shape_overrides: None,
            named_axes: HashMap::new(),
        }
    }

    pub fn set_external_data(&mut self, external: Box<dyn ExternalDataLoader>) {
        self.external = external;
    }

    pub fn set_input_shaper_custom(&mut self, shaper: Box<InputShaper>) {
        self.input_shaper_custom = Some(shaper);
    }

    pub fn force_input_shapes(&mut self, shapes: Vec<Option<Shape>>) {
        self.input_shape_overrides = Some(shapes)
    }

    pub fn add_named_axis(&mut self, name: &str, value: Size) {
        self.named_axes.insert(name.to_owned(), value);
    }

    pub fn load(self) -> OnnxResult<Graph> {
        let mut external = self.external;

        let input_shaper = move |dims: &[OnnxDimValue], name: &str, index| {
            // first try custom shaper
            if let Some(input_shaper_custom) = &self.input_shaper_custom {
                return input_shaper_custom(dims, name, index);
            }
            // then shape overrides
            if let Some(input_shape_overrides) = &self.input_shape_overrides {
                if index < input_shape_overrides.len() {
                    if let Some(shape) = &input_shape_overrides[index] {
                        return Some(shape.clone());
                    }
                } else {
                    return None;
                }
            }
            // finally try basic resolution using named axes
            let mut new_dims = vec![];
            for d in dims {
                let d_new = match *d {
                    OnnxDimValue::Value(value) => Size::fixed(value as usize),
                    OnnxDimValue::Param(ref param) => self.named_axes.get(param)?.clone(),
                };
                new_dims.push(d_new);
            }
            Some(Shape::new(new_dims))
        };

        graph_from_onnx_bytes(self.bytes.deref(), external.as_mut(), &input_shaper)
    }
}