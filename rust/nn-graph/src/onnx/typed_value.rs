use itertools::Itertools;
use unwrap_match::unwrap_match;

use crate::graph::{Graph, Value};
use crate::shape;
use crate::shape::{Shape, Size};

/// A value as it appears in the onnx graph.
///
/// `IntTensor` is stored as a float tensor where casting still needs to happen at use-time.
#[derive(Debug, Clone)]
pub enum TypedValue {
    //TODO support proper tensor types both compile and runtime
    //TODO switch to ::Shape and ::Tensor instead of splitting them here, this is causing a bunch of ugly matches
    Shape(Vec<SizeOrInt>),
    FloatTensor(Value),
    IntTensor(Value),
}

#[derive(Debug, Copy, Clone)]
pub enum SizeOrInt {
    Size(Size),
    Int(i64),
}

impl TypedValue {
    pub fn as_shape(&self, graph: &Graph) -> Option<Vec<SizeOrInt>> {
        match self {
            TypedValue::Shape(shape) => Some(shape.clone()),
            &TypedValue::IntTensor(inner) => {
                let shape_shape = &graph[inner].shape;

                // TODO constrain this again once shapes get proper shapes
                assert!(
                    shape_shape.rank() <= 1,
                    "Shape tensor must have rank 0 or 1, got shape {:?}",
                    shape_shape
                );

                let dims_f = graph.as_const(inner).expect("Shape tensor must be constant");
                let shape = dims_f
                    .iter()
                    .copied()
                    .map(float_to_i64_exact)
                    .map(SizeOrInt::Int)
                    .collect_vec();

                Some(shape)
            }
            TypedValue::FloatTensor(_) => None,
        }
    }

    pub fn unwrap_float(&self) -> Value {
        unwrap_match!(self, &TypedValue::FloatTensor(inner) => inner)
    }

    pub fn unwrap_int(&self) -> Value {
        unwrap_match!(self, &TypedValue::IntTensor(inner) => inner)
    }

    pub fn unwrap_tensor(&self) -> Value {
        match self {
            TypedValue::Shape(_) => panic!("Expected tensor, got {:?}", self),
            &TypedValue::FloatTensor(inner) | &TypedValue::IntTensor(inner) => inner,
        }
    }

    pub fn with_same_type(result: Value, other: &TypedValue) -> TypedValue {
        match other {
            TypedValue::Shape(_) => panic!("Cannot wrap value as shape"),
            TypedValue::FloatTensor(_) => TypedValue::FloatTensor(result),
            TypedValue::IntTensor(_) => TypedValue::IntTensor(result),
        }
    }

    pub fn shape(&self, graph: &Graph) -> Shape {
        match self {
            TypedValue::Shape(shape) => shape![shape.len()],
            &TypedValue::FloatTensor(tensor) | &TypedValue::IntTensor(tensor) => graph[tensor].shape.clone(),
        }
    }
}

pub fn float_to_i64_exact(f: f32) -> i64 {
    assert_eq!(f as i64 as f32, f, "Float must be an integer, got {}", f);
    f as i64
}

impl SizeOrInt {
    pub fn as_size(self) -> Option<Size> {
        match self {
            SizeOrInt::Size(size) => Some(size),
            SizeOrInt::Int(int) => {
                if int >= 0 {
                    Some(Size::fixed(int as usize))
                } else {
                    None
                }
            }
        }
    }
}
