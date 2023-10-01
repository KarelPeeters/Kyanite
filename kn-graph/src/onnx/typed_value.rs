use std::fmt::{Display, Formatter};
use itertools::Itertools;
use num_traits::Signed;

use crate::dtype::{DTensor, DType, Tensor};
use crate::graph::{Graph, Value};
use crate::shape::{ConcreteShape, Shape, Size};

// TODO find a good name
/// A value as it appears in the onnx graph.
///
/// Values are stored as `DTensor` whenever possible,
/// and only resort to `Shape` when elements using [Size::BATCH] are present.
#[derive(Debug, Clone)]
pub enum OnnxValue {
    Value(Value),
    Size(Tensor<SignedSize>),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct SignedSize {
    batch_exp: u32,
    fixed_factor: i64,
}

#[derive(Debug)]
pub struct Overflow;

#[derive(Debug)]
pub enum AsShapeError {
    NonConstant,
    WrongType { expected: DType, actual: DType },
    WrongShape { shape: ConcreteShape },
    Overflow(Overflow),
}

impl OnnxValue {
    pub fn new_size(tensor: Tensor<SignedSize>, graph: &mut Graph) -> Self {
        if tensor.iter().any(|x| x.batch_exp != 0) {
            OnnxValue::Size(tensor)
        } else {
            let tensor = tensor.map(|x| x.unwrap_fixed().unwrap());
            let value = graph.constant_tensor(DTensor::I64(tensor.into_shared()));
            OnnxValue::Value(value)
        }
    }

    pub fn unwrap_value(&self) -> Option<Value> {
        match self {
            &OnnxValue::Value(value) => Some(value),
            OnnxValue::Size(_) => None,
        }
    }

    pub fn as_size(&self, graph: &Graph) -> Result<Tensor<SignedSize>, AsShapeError> {
        match self {
            &OnnxValue::Value(value) => {
                let value = graph.as_const(value).ok_or(AsShapeError::NonConstant)?;
                if let DTensor::I64(value) = value {
                    Ok(value.mapv(SignedSize::from_int).into_shared())
                } else {
                    return Err(AsShapeError::WrongType { expected: DType::I64, actual: value.dtype() });
                }
            }
            OnnxValue::Size(size) => Ok(size.clone()),
        }
    }

    pub fn as_signed_shape(&self, graph: &Graph) -> Result<Vec<SignedSize>, AsShapeError> {
        let shape_tensor = self.as_size(graph)?;

        if shape_tensor.shape().len() != 1 {
            return Err(AsShapeError::WrongShape { shape: ConcreteShape::new(shape_tensor.shape().to_vec()) });
        }

        Ok(shape_tensor.iter().copied().collect_vec())
    }

    pub fn as_shape(&self, graph: &Graph) -> Result<Shape, AsShapeError> {
        let signed = self.as_signed_shape(graph)?;
        let unsigned = signed.iter()
            .map(|v| v.to_size())
            .try_collect()
            .map_err(|e| AsShapeError::Overflow(e))?;
        Ok(Shape::new(unsigned))
    }


    pub fn dtype(&self, graph: &Graph) -> DType {
        match self {
            &OnnxValue::Value(value) => graph[value].dtype,
            OnnxValue::Size(_) => DType::I64,
        }
    }

    pub fn shape(&self, graph: &Graph) -> Shape {
        match self {
            &OnnxValue::Value(value) => graph[value].shape.clone(),
            OnnxValue::Size(size) => Shape::fixed(size.shape()),
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
    pub const BATCH: SignedSize = SignedSize { batch_exp: 1, fixed_factor: 1 };
}

impl SignedSize {
    pub const fn new(batch_exp: u32, fixed_factor: i64) -> Self {
        let batch_exp = if fixed_factor == 0 {
            0
        } else {
            batch_exp
        };

        SignedSize { batch_exp, fixed_factor }
    }

    pub const fn from_int(i: i64) -> SignedSize {
        SignedSize::new(0, i)
    }

    pub fn from_size(size: Size) -> Result<SignedSize, Overflow> {
        let (factor, exp) = size.components_factor_exp();
        let factor: i64 = factor.try_into().map_err(|_| Overflow)?;
        Ok(SignedSize::new(exp, factor))
    }

    pub fn to_size(self) -> Result<Size, Overflow> {
        let factor: usize = self.fixed_factor.try_into().map_err(|_| Overflow)?;
        Ok(Size::new(self.batch_exp, factor))
    }

    pub fn floor_div(self, rhs: Self) -> Option<Self> {
        if self.batch_exp < rhs.batch_exp {
            None
        } else {
            Some(SignedSize::new(
                self.batch_exp - rhs.batch_exp,
                self.fixed_factor / rhs.fixed_factor,
            ))
        }
    }

    pub fn unwrap_fixed(self) -> Option<i64> {
        if self.batch_exp == 0 {
            Some(self.fixed_factor)
        } else {
            None
        }
    }
}

impl Display for SignedSize {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match (self.fixed_factor, self.batch_exp) {
            (a, 0) => write!(f, "{}", a),
            (1, 1) => write!(f, "B"),
            (a, 1) => write!(f, "{}B", a),
            (1, b) => write!(f, "B^{}", b),
            (a, b) => write!(f, "{}B^{}", a, b),
        }
    }
}

impl std::ops::Neg for SignedSize {
    type Output = Self;

    fn neg(self) -> Self::Output {
        SignedSize::new(self.batch_exp, -self.fixed_factor)
    }
}

impl std::ops::Add for SignedSize {
    type Output = Option<Self>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.batch_exp == rhs.batch_exp {
            Some(SignedSize::new(self.batch_exp, self.fixed_factor + rhs.fixed_factor))
        } else {
            None
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
        SignedSize::new(self.batch_exp + rhs.batch_exp, self.fixed_factor * rhs.fixed_factor)
    }
}

impl std::ops::Div for SignedSize {
    type Output = Option<Self>;

    // only returns [Some] for exact division
    fn div(self, rhs: Self) -> Self::Output {
        if self.batch_exp < rhs.batch_exp || self.fixed_factor % rhs.fixed_factor != 0 {
            None
        } else {
            Some(SignedSize::new(
                self.batch_exp - rhs.batch_exp,
                self.fixed_factor / rhs.fixed_factor,
            ))
        }
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
