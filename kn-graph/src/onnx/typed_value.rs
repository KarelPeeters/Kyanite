use std::fmt::{Display, Formatter};

use itertools::Itertools;
use ndarray::{ArcArray, IxDyn};
use unwrap_match::unwrap_match;

use crate::graph::{Graph, Value};
use crate::onnx::result::{Node, OnnxError, OnnxResult};
use crate::shape;
use crate::shape::{Shape, Size};

/// A value as it appears in the onnx graph.
///
/// `IntTensor` is stored as a float tensor where casting still needs to happen at use-time.
#[derive(Debug, Clone)]
pub enum TypedValue {
    //TODO support proper tensor types both compile and runtime
    //TODO switch to ::Shape and ::Tensor instead of splitting them here, this is causing a bunch of ugly matches
    Shape(Vec<SignedSize>),
    FloatTensor(Value),
    IntTensor(Value),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct SignedSize {
    negative: bool,
    size: Size,
}

impl TypedValue {
    pub fn as_partial_shape(&self, graph: &Graph) -> Option<Vec<SignedSize>> {
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

                let dims_f = graph.as_const(inner)?;
                let shape = dims_f
                    .iter()
                    .copied()
                    .map(float_to_i64_exact)
                    .map(SignedSize::from_int)
                    .collect_vec();

                Some(shape)
            }
            TypedValue::FloatTensor(_) => None,
        }
    }

    pub fn as_shape(&self, graph: &Graph) -> Option<Shape> {
        let partial = self.as_partial_shape(graph)?;
        let dims = partial.iter().map(|s| s.as_size()).collect::<Option<_>>()?;
        Some(Shape::new(dims))
    }

    pub fn unwrap_partial_shape(&self, node: Node<&str>, graph: &Graph) -> OnnxResult<Vec<SignedSize>> {
        self.as_partial_shape(graph)
            .ok_or_else(|| OnnxError::UnsupportedPartialShape(node.to_owned(), format!("{:?}", self)))
    }

    pub fn unwrap_shape(&self, node: Node<&str>, graph: &Graph) -> OnnxResult<Shape> {
        self.as_shape(graph)
            .ok_or_else(|| OnnxError::UnsupportedShape(node.to_owned(), format!("{:?}", self)))
    }

    pub fn unwrap_float(&self) -> Value {
        unwrap_match!(self, &TypedValue::FloatTensor(inner) => inner)
    }

    pub fn unwrap_int(&self, graph: &mut Graph) -> Value {
        match self {
            TypedValue::Shape(shape) => {
                let data = shape
                    .iter()
                    .map(|s| {
                        let sign = if s.negative { -1.0 } else { 1.0 };
                        let size = s.size.unwrap_fixed("int tensor value") as f32;
                        sign * size
                    })
                    .collect_vec();

                graph.constant(shape![shape.len()], data)
            }
            TypedValue::FloatTensor(_) => panic!("Expected int tensor, got float"),
            &TypedValue::IntTensor(value) => value,
        }
    }

    pub fn unwrap_tensor(&self) -> Value {
        match self {
            TypedValue::Shape(_) => panic!("Expected tensor, got {:?}", self),
            &TypedValue::FloatTensor(inner) | &TypedValue::IntTensor(inner) => inner,
        }
    }

    pub fn unwrap_const_int(&self, graph: &mut Graph) -> ArcArray<i64, IxDyn> {
        let value = self.unwrap_int(graph);
        graph.as_const(value).unwrap().mapv(float_to_i64_exact).to_shared()
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

#[allow(dead_code)]
impl SignedSize {
    pub const ZERO: SignedSize = SignedSize::from_int(0);
    pub const ONE: SignedSize = SignedSize::from_int(1);
    pub const NEG_ONE: SignedSize = SignedSize::from_int(-1);
    pub const BATCH: SignedSize = SignedSize::new(false, Size::BATCH);
}

impl SignedSize {
    pub const fn new(negative: bool, size: Size) -> Self {
        if size.is_zero() {
            SignedSize { negative: false, size }
        } else {
            SignedSize { negative, size }
        }
    }

    pub const fn from_int(i: i64) -> SignedSize {
        SignedSize::new(i < 0, Size::fixed(i.abs_diff(0) as usize))
    }

    pub const fn as_size(self) -> Option<Size> {
        if self.negative {
            None
        } else {
            Some(self.size)
        }
    }

    pub fn floor_div(self, rhs: Self) -> Option<Self> {
        Some(SignedSize::new(
            self.negative ^ rhs.negative,
            self.size.floor_div(rhs.size)?,
        ))
    }

    pub fn is_zero(&self) -> bool {
        self.size.is_zero()
    }
}

impl Display for SignedSize {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let sign = if self.negative { "-" } else { "" };
        write!(f, "{}{}", sign, self.size)
    }
}

impl std::ops::Neg for SignedSize {
    type Output = Self;

    fn neg(self) -> Self::Output {
        // constructor will normalize the sign for size 0
        SignedSize::new(!self.negative, self.size)
    }
}

impl From<Size> for SignedSize {
    fn from(value: Size) -> Self {
        SignedSize::new(false,  value)
    }
}

impl std::ops::Add for SignedSize {
    type Output = Option<Self>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self.negative, rhs.negative) {
            (false, false) => Some(SignedSize::from( (self.size + rhs.size)?)),
            (true, true) => Some(SignedSize::new(true, (self.size + rhs.size)?)),
            (false, true) | (true, false) => {
                let flip0 = self.negative ^ rhs.negative;
                let (flip1, size) = match ((self.size - rhs.size), (rhs.size - self.size)) {
                    (Some(size), _) => Some((false, size)),
                    (_, Some(size)) => Some((true, size)),
                    (None, None) => None,
                }?;
                Some(SignedSize::new(!(flip0 ^ flip1), size))
            }
        }
    }
}

impl std::ops::Sub for SignedSize {
    type Output = Option<Self>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Mul for SignedSize {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        SignedSize::new(self.negative ^ rhs.negative, self.size * rhs.size)
    }
}

impl std::ops::Div for SignedSize {
    type Output = Option<Self>;

    fn div(self, rhs: Self) -> Self::Output {
        Some(SignedSize::new(self.negative ^ rhs.negative, (self.size / rhs.size)?))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn add() {
        assert_eq!(SignedSize::BATCH + SignedSize::ONE, None);
        assert_eq!(
            SignedSize::from_int(3) - SignedSize::from_int(2),
            Some(SignedSize::from_int(1))
        );
        assert_eq!(
            SignedSize::from_int(3) - SignedSize::from_int(4),
            Some(SignedSize::from_int(-1))
        );
        assert_eq!(
            SignedSize::from_int(3) + -SignedSize::from_int(4),
            Some(SignedSize::from_int(-1))
        );
        assert_eq!(
            SignedSize::from_int(3) - -SignedSize::from_int(4),
            Some(SignedSize::from_int(7))
        );

        assert_eq!(SignedSize::ZERO - SignedSize::BATCH, Some(-SignedSize::BATCH))
    }
}
