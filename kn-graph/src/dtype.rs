use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::num::FpCategory;
use std::ops::Deref;

use decorum::cmp::FloatEq;
use decorum::hash::FloatHash;
use itertools::zip_eq;
use ndarray::{ArcArray, IntoDimension, IxDyn, LinalgScalar};

#[derive(Debug, Copy, Clone)]
pub struct T32(pub f32);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum DType {
    F32,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    // TODO add bool and f64
    // Bool,
    // F64
}

pub type Tensor<T> = ArcArray<T, IxDyn>;

#[derive(Debug, Clone)]
pub enum DTensor {
    F32(Tensor<f32>),
    I8(Tensor<i8>),
    I16(Tensor<i16>),
    I32(Tensor<i32>),
    I64(Tensor<i64>),
    U8(Tensor<u8>),
    U16(Tensor<u16>),
    U32(Tensor<u32>),
    U64(Tensor<u64>),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum DScalar {
    F32(T32),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum DSize {
    S8,
    S16,
    S32,
    S64,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Specials {
    pub zero: DScalar,
    pub one: DScalar,
    pub min: DScalar,
    pub max: DScalar,
}

impl DType {
    pub fn size(self) -> DSize {
        match self {
            DType::F32 => DSize::S32,
            DType::U8 | DType::I8 => DSize::S8,
            DType::U16 | DType::I16 => DSize::S16,
            DType::U32 | DType::I32 => DSize::S32,
            DType::U64 | DType::I64 => DSize::S64,
        }
    }

    pub fn is_signed(self) -> bool {
        match self {
            DType::F32 => true,
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => true,
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => false,
        }
    }

    pub fn is_float(self) -> bool {
        match self {
            DType::F32 => true,
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => false,
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => false,
        }
    }

    pub fn is_int(self) -> bool {
        match self {
            DType::F32 => false,
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => true,
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => true,
        }
    }

    pub fn specials(self) -> Specials {
        match self {
            DType::F32 => Specials::new(f32::NEG_INFINITY, f32::INFINITY),
            DType::I8 => Specials::new(i8::MIN, i8::MAX),
            DType::I16 => Specials::new(i16::MIN, i16::MAX),
            DType::I32 => Specials::new(i32::MIN, i32::MAX),
            DType::I64 => Specials::new(i64::MIN, i64::MAX),
            DType::U8 => Specials::new(u8::MIN, u8::MAX),
            DType::U16 => Specials::new(u16::MIN, u16::MAX),
            DType::U32 => Specials::new(u32::MIN, u32::MAX),
            DType::U64 => Specials::new(u64::MIN, u64::MAX),
        }
    }

    pub fn as_c_str(self) -> &'static str {
        match self {
            DType::F32 => "float",
            DType::I8 => "int8_t",
            DType::I16 => "int16_t",
            DType::I32 => "int32_t",
            DType::I64 => "int64_t",
            DType::U8 => "uint8_t",
            DType::U16 => "uint16_t",
            DType::U32 => "uint32_t",
            DType::U64 => "uint64_t",
        }
    }
}

impl DSize {
    pub fn bytes(self) -> usize {
        match self {
            DSize::S8 => 1,
            DSize::S16 => 2,
            DSize::S32 => 4,
            DSize::S64 => 8,
        }
    }
}

impl DScalar {
    pub fn f32(x: f32) -> Self {
        DScalar::F32(T32(x))
    }

    pub fn dtype(self) -> DType {
        match self {
            DScalar::F32(_) => DType::F32,
            DScalar::I8(_) => DType::I8,
            DScalar::I16(_) => DType::I16,
            DScalar::I32(_) => DType::I32,
            DScalar::I64(_) => DType::I64,
            DScalar::U8(_) => DType::U8,
            DScalar::U16(_) => DType::U16,
            DScalar::U32(_) => DType::U32,
            DScalar::U64(_) => DType::U64,
        }
    }

    pub fn unwrap_f32(self) -> Option<f32> {
        match self {
            DScalar::F32(x) => Some(x.0),
            _ => None,
        }
    }

    pub fn to_tensor(self) -> DTensor {
        match self {
            DScalar::F32(T32(s)) => DTensor::F32(ArcArray::from_shape_vec(IxDyn(&[]), vec![s]).unwrap()),
            DScalar::I8(x) => DTensor::I8(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
            DScalar::I16(x) => DTensor::I16(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
            DScalar::I32(x) => DTensor::I32(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
            DScalar::I64(x) => DTensor::I64(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
            DScalar::U8(x) => DTensor::U8(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
            DScalar::U16(x) => DTensor::U16(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
            DScalar::U32(x) => DTensor::U32(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
            DScalar::U64(x) => DTensor::U64(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
        }
    }

    pub fn unwrap_uint(self) -> Option<u64> {
        match self {
            DScalar::U8(x) => Some(x as u64),
            DScalar::U16(x) => Some(x as u64),
            DScalar::U32(x) => Some(x as u64),
            DScalar::U64(x) => Some(x),
            _ => None,
        }
    }

    pub fn to_c_str(self) -> String {
        match self {
            DScalar::F32(c) => DisplayCFloat(*c).to_string(),
            DScalar::U8(c) => format!("{}", c),
            DScalar::U16(c) => format!("{}", c),
            DScalar::U32(c) => format!("{}", c),
            DScalar::U64(c) => format!("{}", c),
            DScalar::I8(c) => format!("{}", c),
            DScalar::I16(c) => format!("{}", c),
            DScalar::I32(c) => format!("{}", c),
            DScalar::I64(c) => format!("{}", c),
        }
    }

    pub fn value_cast(self, to: DType) -> DScalar {
        // cast to big general value
        let (yf, yi) = match self {
            DScalar::F32(T32(x)) => (x as f64, x as i128),
            DScalar::I8(x) => (x as f64, x as i128),
            DScalar::I16(x) => (x as f64, x as i128),
            DScalar::I32(x) => (x as f64, x as i128),
            DScalar::I64(x) => (x as f64, x as i128),
            DScalar::U8(x) => (x as f64, x as i128),
            DScalar::U16(x) => (x as f64, x as i128),
            DScalar::U32(x) => (x as f64, x as i128),
            DScalar::U64(x) => (x as f64, x as i128),
        };

        // convert to target
        match to {
            DType::F32 => DScalar::f32(yf as f32),
            DType::I8 => DScalar::I8(yi as i8),
            DType::I16 => DScalar::I16(yi as i16),
            DType::I32 => DScalar::I32(yi as i32),
            DType::I64 => DScalar::I64(yi as i64),
            DType::U8 => DScalar::U8(yi as u8),
            DType::U16 => DScalar::U16(yi as u16),
            DType::U32 => DScalar::U32(yi as u32),
            DType::U64 => DScalar::U64(yi as u64),
        }
    }

    pub fn bit_cast(self, to: DType) -> Option<DScalar> {
        if self.dtype().size() != to.size() {
            return None;
        }

        // convert to bits, zero-extend just to be safe
        let bits = match self {
            DScalar::F32(T32(x)) => x.to_bits() as u64,
            DScalar::I8(x) => x as u8 as u64,
            DScalar::I16(x) => x as u16 as u64,
            DScalar::I32(x) => x as u32 as u64,
            DScalar::I64(x) => x as u64,
            DScalar::U8(x) => x as u64,
            DScalar::U16(x) => x as u64,
            DScalar::U32(x) => x as u64,
            DScalar::U64(x) => x,
        };

        // convert to target
        let y = match to {
            DType::F32 => DScalar::f32(f32::from_bits(bits as u32)),
            DType::I8 => DScalar::I8(bits as i8),
            DType::I16 => DScalar::I16(bits as i16),
            DType::I32 => DScalar::I32(bits as i32),
            DType::I64 => DScalar::I64(bits as i64),
            DType::U8 => DScalar::U8(bits as u8),
            DType::U16 => DScalar::U16(bits as u16),
            DType::U32 => DScalar::U32(bits as u32),
            DType::U64 => DScalar::U64(bits),
        };

        Some(y)
    }
}

pub trait IntoDScalar: LinalgScalar + PartialEq {
    const DTYPE: DType;
    fn to_dscalar(&self) -> DScalar;
    fn from_dscalar(scalar: DScalar) -> Option<Self>;
    fn vec_to_dtensor(data: Vec<Self>) -> DTensor;
}

macro_rules! impl_into_dscalar {
    ($ty:ty, $dtype:expr, $dtensor:ident, |$x:ident| $conv:expr, $pattern:pat => $result:expr) => {
        impl IntoDScalar for $ty {
            const DTYPE: DType = $dtype;

            fn to_dscalar(&self) -> DScalar {
                let &$x = self;
                $conv
            }

            fn from_dscalar(scalar: DScalar) -> Option<Self> {
                match scalar {
                    $pattern => Some($result),
                    _ => None,
                }
            }

            fn vec_to_dtensor(data: Vec<Self>) -> DTensor {
                DTensor::$dtensor(ArcArray::from_vec(data).into_dyn())
            }
        }
    };
}

impl_into_dscalar!(f32, DType::F32, F32, |x| DScalar::f32(x), DScalar::F32(T32(x)) => x);
impl_into_dscalar!(i8, DType::I8, I8, |x| DScalar::I8(x), DScalar::I8(x) => x);
impl_into_dscalar!(i16, DType::I16, I16, |x| DScalar::I16(x), DScalar::I16(x) => x);
impl_into_dscalar!(i32, DType::I32, I32, |x| DScalar::I32(x), DScalar::I32(x) => x);
impl_into_dscalar!(i64, DType::I64, I64, |x| DScalar::I64(x), DScalar::I64(x) => x);
impl_into_dscalar!(u8, DType::U8, U8, |x| DScalar::U8(x), DScalar::U8(x) => x);
impl_into_dscalar!(u16, DType::U16, U16, |x| DScalar::U16(x), DScalar::U16(x) => x);
impl_into_dscalar!(u32, DType::U32, U32, |x| DScalar::U32(x), DScalar::U32(x) => x);
impl_into_dscalar!(u64, DType::U64, U64, |x| DScalar::U64(x), DScalar::U64(x) => x);

#[rustfmt::skip]
#[macro_export]
macro_rules! dispatch_dtensor {
    ($outer:expr, |$ty:ident, $f:ident, $inner:ident| $expr:expr) => {{
        use $crate::dtype::DTensor;
        match $outer {
            DTensor::F32($inner) => { type $ty=f32; let $f=DTensor::F32; { $expr } },
            DTensor::I8($inner) => { type $ty=i8; let $f=DTensor::I8; { $expr } },
            DTensor::I16($inner) => { type $ty=i16; let $f=DTensor::I16; { $expr } },
            DTensor::I32($inner) => { type $ty=i32; let $f=DTensor::I32; { $expr } },
            DTensor::I64($inner) => { type $ty=i64; let $f=DTensor::I64; { $expr } },
            DTensor::U8($inner) => { type $ty=u8; let $f=DTensor::U8; { $expr } },
            DTensor::U16($inner) => { type $ty=u16; let $f=DTensor::U16; { $expr } },
            DTensor::U32($inner) => { type $ty=u32; let $f=DTensor::U32; { $expr } },
            DTensor::U64($inner) => { type $ty=u64; let $f=DTensor::U64; { $expr } },
        }
    }};
}

#[rustfmt::skip]
#[macro_export]
macro_rules! dispatch_dtensor_pair {
    ($out_left:expr, $out_right:expr, |$ty:ident, $f:ident, $in_left:ident, $in_right:ident| $expr:expr) => {{
        use $crate::dtype::DTensor;

        let out_left = $out_left;
        let out_right = $out_right;
        let dtype_left = out_left.dtype();
        let dtype_right = out_right.dtype();
        
        match (out_left, out_right) {
            (DTensor::F32($in_left), DTensor::F32($in_right)) => { type $ty=f32; let $f=DTensor::F32; { $expr } },
            (DTensor::I8($in_left), DTensor::I8($in_right)) => { type $ty=i8; let $f=DTensor::I8; { $expr } },
            (DTensor::I16($in_left), DTensor::I16($in_right)) => { type $ty=i16; let $f=DTensor::I16; { $expr } },
            (DTensor::I32($in_left), DTensor::I32($in_right)) => { type $ty=i32; let $f=DTensor::I32; { $expr } },
            (DTensor::I64($in_left), DTensor::I64($in_right)) => { type $ty=i64; let $f=DTensor::I64; { $expr } },
            (DTensor::U8($in_left), DTensor::U8($in_right)) => { type $ty=u8; let $f=DTensor::U8; { $expr } },
            (DTensor::U16($in_left), DTensor::U16($in_right)) => { type $ty=u16; let $f=DTensor::U16; { $expr } },
            (DTensor::U32($in_left), DTensor::U32($in_right)) => { type $ty=u32; let $f=DTensor::U32; { $expr } },
            (DTensor::U64($in_left), DTensor::U64($in_right)) => { type $ty=u64; let $f=DTensor::U64; { $expr } },
            _ => panic!("Mismatched dtypes: left {:?}, right {:?}", dtype_left, dtype_right),
        }
    }};
}

#[macro_export]
macro_rules! map_dtensor {
    ($outer:expr, |$inner:ident| $expr:expr) => {
        crate::dtype::dispatch_dtensor!($outer, |_T, f, $inner| f($expr))
    };
}

#[macro_export]
macro_rules! map_dtensor_pair {
    ($out_left:expr, $out_right:expr, |$in_left:ident, $in_right:ident| $expr:expr) => {
        crate::dtype::dispatch_dtensor_pair!($out_left, $out_right, |_T, f, $in_left, $in_right| f($expr))
    };
}

#[rustfmt::skip]
#[macro_export]
macro_rules! map_dscalar_pair {
    ($out_left:expr, $out_right:expr, |$in_left:ident, $in_right:ident| $expr:expr) => {{
        use crate::dtype::{DScalar, T32};
        
        let out_left = $out_left;
        let out_right = $out_right;
        
        match (out_left, out_right) {
            (DScalar::F32(T32($in_left)), DScalar::F32(T32($in_right))) => DScalar::F32(T32($expr)),
            (DScalar::I8($in_left), DScalar::I8($in_right)) => DScalar::I8($expr),
            (DScalar::I16($in_left), DScalar::I16($in_right)) => DScalar::I16($expr),
            (DScalar::I32($in_left), DScalar::I32($in_right)) => DScalar::I32($expr),
            (DScalar::I64($in_left), DScalar::I64($in_right)) => DScalar::I64($expr),
            (DScalar::U8($in_left), DScalar::U8($in_right)) => DScalar::U8($expr),
            (DScalar::U16($in_left), DScalar::U16($in_right)) => DScalar::U16($expr),
            (DScalar::U32($in_left), DScalar::U32($in_right)) => DScalar::U32($expr),
            (DScalar::U64($in_left), DScalar::U64($in_right)) => DScalar::U64($expr),
            _ => panic!("Mismatched dtypes: left {:?}, right {:?}", out_left, out_right),
        }
    }}
}

// export macros
pub use dispatch_dtensor;
pub use dispatch_dtensor_pair;
pub use map_dscalar_pair;
pub use map_dtensor;
pub use map_dtensor_pair;

impl DTensor {
    pub fn shape(&self) -> &[usize] {
        dispatch_dtensor!(self, |_T, _f, inner| inner.shape())
    }

    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    pub fn len(&self) -> usize {
        self.shape().iter().copied().product()
    }

    pub fn dtype(&self) -> DType {
        dispatch_dtensor!(self, |T, _f, _i| T::DTYPE)
    }

    pub fn reshape<E: IntoDimension>(&self, shape: E) -> DTensor {
        map_dtensor!(self, |inner| inner.reshape(shape).into_dyn())
    }

    // TODO generic unwrap function?
    pub fn unwrap_f32(&self) -> Option<&Tensor<f32>> {
        match self {
            DTensor::F32(tensor) => Some(tensor),
            _ => None,
        }
    }

    pub fn unwrap_i64(&self) -> Option<&Tensor<i64>> {
        match self {
            DTensor::I64(tensor) => Some(tensor),
            _ => None,
        }
    }
}

impl Eq for DTensor {}

impl PartialEq for DTensor {
    fn eq(&self, other: &Self) -> bool {
        if self.shape() != other.shape() || self.dtype() != other.dtype() {
            return false;
        }

        match (self, other) {
            // proper float compare
            (DTensor::F32(a), DTensor::F32(b)) => zip_eq(a.iter(), b.iter()).all(|(a, b)| a.float_eq(b)),

            // ints can be compared like normal
            (DTensor::I8(a), DTensor::I8(b)) => a == b,
            (DTensor::I16(a), DTensor::I16(b)) => a == b,
            (DTensor::I32(a), DTensor::I32(b)) => a == b,
            (DTensor::I64(a), DTensor::I64(b)) => a == b,
            (DTensor::U8(a), DTensor::U8(b)) => a == b,
            (DTensor::U16(a), DTensor::U16(b)) => a == b,
            (DTensor::U32(a), DTensor::U32(b)) => a == b,
            (DTensor::U64(a), DTensor::U64(b)) => a == b,
            _ => unreachable!(),
        }
    }
}

impl Deref for T32 {
    type Target = f32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq<Self> for T32 {
    fn eq(&self, other: &Self) -> bool {
        self.0.float_eq(&other.0)
    }
}

impl Eq for T32 {}

impl Hash for T32 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.float_hash(state)
    }
}

impl Specials {
    pub fn new<T: IntoDScalar + num_traits::Zero + num_traits::One>(min: T, max: T) -> Self {
        Self {
            zero: T::zero().to_dscalar(),
            one: T::one().to_dscalar(),
            min: min.to_dscalar(),
            max: max.to_dscalar(),
        }
    }
}

#[derive(Debug)]
pub struct DisplayCFloat(pub f32);

impl Display for DisplayCFloat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = if self.0.is_sign_negative() { "-" } else { "" };

        match self.0.classify() {
            FpCategory::Nan => write!(f, "({s}(0.0/0.0))"),
            FpCategory::Infinite => write!(f, "({s}(1.0/0.0))"),
            FpCategory::Zero => write!(f, "({s}0.0)"),
            FpCategory::Subnormal | FpCategory::Normal => write!(f, "{}", self.0),
        }
    }
}
