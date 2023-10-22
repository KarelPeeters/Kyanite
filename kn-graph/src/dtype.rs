use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::num::FpCategory;
use std::ops::Deref;

use bytemuck::NoUninit;
use decorum::cmp::FloatEq;
use decorum::hash::FloatHash;
use itertools::zip_eq;
use ndarray::{ArcArray, IntoDimension, IxDyn, LinalgScalar};

#[derive(Debug, Copy, Clone)]
pub struct T32(pub f32);

#[derive(Debug, Copy, Clone)]
pub struct T64(pub f64);

// TODO maybe remove this at some point and switch to proper bools,
//   and figure out another solution to the "bool arithmetic" problem
#[derive(Debug, Copy, Clone, Eq, Ord, PartialOrd, PartialEq, Hash)]
#[repr(transparent)]
pub struct DBool(pub bool);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum DType {
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

pub type Tensor<T> = ArcArray<T, IxDyn>;

#[derive(Debug, Clone)]
pub enum DTensor {
    F32(Tensor<f32>),
    F64(Tensor<f64>),
    I8(Tensor<i8>),
    I16(Tensor<i16>),
    I32(Tensor<i32>),
    I64(Tensor<i64>),
    U8(Tensor<u8>),
    U16(Tensor<u16>),
    U32(Tensor<u32>),
    U64(Tensor<u64>),
    Bool(Tensor<DBool>),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum DScalar {
    F32(T32),
    F64(T64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    Bool(DBool),
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct DInfo {
    pub size: DSize,
    pub signed: bool,
    pub float: bool,
    pub int: bool,
    pub is_bool: bool,
}

impl DType {
    pub fn info(self) -> DInfo {
        match self {
            DType::F32 => DInfo::float(DSize::S32),
            DType::F64 => DInfo::float(DSize::S64),
            DType::I8 => DInfo::int(DSize::S8, true),
            DType::I16 => DInfo::int(DSize::S16, true),
            DType::I32 => DInfo::int(DSize::S32, true),
            DType::I64 => DInfo::int(DSize::S64, true),
            DType::U8 => DInfo::int(DSize::S8, false),
            DType::U16 => DInfo::int(DSize::S16, false),
            DType::U32 => DInfo::int(DSize::S32, false),
            DType::U64 => DInfo::int(DSize::S64, false),
            DType::Bool => DInfo::bool(),
        }
    }

    pub fn size(self) -> DSize {
        self.info().size
    }

    pub fn is_signed(self) -> bool {
        self.info().signed
    }

    pub fn is_float(self) -> bool {
        self.info().float
    }

    pub fn is_int(self) -> bool {
        self.info().int
    }

    pub fn is_bool(self) -> bool {
        self.info().is_bool
    }

    // TODO move specials to type itself, while keeping this one too?
    pub fn specials(self) -> Specials {
        match self {
            DType::F32 => Specials::new(f32::NEG_INFINITY, f32::INFINITY),
            DType::F64 => Specials::new(f64::NEG_INFINITY, f64::INFINITY),
            DType::I8 => Specials::new(i8::MIN, i8::MAX),
            DType::I16 => Specials::new(i16::MIN, i16::MAX),
            DType::I32 => Specials::new(i32::MIN, i32::MAX),
            DType::I64 => Specials::new(i64::MIN, i64::MAX),
            DType::U8 => Specials::new(u8::MIN, u8::MAX),
            DType::U16 => Specials::new(u16::MIN, u16::MAX),
            DType::U32 => Specials::new(u32::MIN, u32::MAX),
            DType::U64 => Specials::new(u64::MIN, u64::MAX),
            DType::Bool => Specials::new(DBool(false), DBool(true)),
        }
    }

    pub fn as_c_str(self) -> &'static str {
        match self {
            DType::F32 => "float",
            DType::F64 => "double",
            DType::I8 => "int8_t",
            DType::I16 => "int16_t",
            DType::I32 => "int32_t",
            DType::I64 => "int64_t",
            DType::U8 => "uint8_t",
            DType::U16 => "uint16_t",
            DType::U32 => "uint32_t",
            DType::U64 => "uint64_t",
            DType::Bool => "bool",
        }
    }
}

impl DInfo {
    fn int(size: DSize, signed: bool) -> Self {
        DInfo {
            size,
            signed,
            float: false,
            int: true,
            is_bool: false,
        }
    }

    fn float(size: DSize) -> Self {
        DInfo {
            size,
            signed: true,
            float: true,
            int: false,
            is_bool: false,
        }
    }

    fn bool() -> Self {
        DInfo {
            size: DSize::S8,
            signed: false,
            float: false,
            int: false,
            is_bool: true,
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

#[rustfmt::skip]
#[macro_export]
macro_rules! dispatch_dtype {
    ($outer:expr, |$ty:ident, $fs:ident, $ft:ident| $expr:expr) => {{
        use $crate::dtype::{DType, DBool, DScalar, DTensor};
        match $outer {
            DType::F32 => { type $ty=f32; let $fs=DScalar::F32; let $ft=DTensor::F32; { $expr } }
            DType::F64 => { type $ty=f64; let $fs=DScalar::F64; let $ft=DTensor::F64; { $expr } }
            DType::I8 => { type $ty=i8; let $fs=DScalar::I8; let $ft=DTensor::I8; { $expr } }
            DType::I16 => { type $ty=i16; let $fs=DScalar::I16; let $ft=DTensor::I16; { $expr } }
            DType::I32 => { type $ty=i32; let $fs=DScalar::I32; let $ft=DTensor::I32; { $expr } }
            DType::I64 => { type $ty=i64; let $fs=DScalar::I64; let $ft=DTensor::I64; { $expr } }
            DType::U8 => { type $ty=u8; let $fs=DScalar::U8; let $ft=DTensor::U8; { $expr } }
            DType::U16 => { type $ty=u16; let $fs=DScalar::U16; let $ft=DTensor::U16; { $expr } }
            DType::U32 => { type $ty=u32; let $fs=DScalar::U32; let $ft=DTensor::U32; { $expr } }
            DType::U64 => { type $ty=u64; let $fs=DScalar::U64; let $ft=DTensor::U64; { $expr } }
            DType::Bool => { type $ty=DBool; let $fs=DScalar::Bool; let $ft=DTensor::Bool; { $expr } }
        }
    }};
}

impl DScalar {
    pub fn f32(x: f32) -> Self {
        DScalar::F32(T32(x))
    }

    pub fn f64(x: f64) -> Self {
        DScalar::F64(T64(x))
    }

    pub fn bool(x: bool) -> Self {
        DScalar::Bool(DBool(x))
    }

    pub fn dtype(self) -> DType {
        match self {
            DScalar::F32(_) => DType::F32,
            DScalar::F64(_) => DType::F64,
            DScalar::I8(_) => DType::I8,
            DScalar::I16(_) => DType::I16,
            DScalar::I32(_) => DType::I32,
            DScalar::I64(_) => DType::I64,
            DScalar::U8(_) => DType::U8,
            DScalar::U16(_) => DType::U16,
            DScalar::U32(_) => DType::U32,
            DScalar::U64(_) => DType::U64,
            DScalar::Bool(_) => DType::Bool,
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
            DScalar::F64(T64(s)) => DTensor::F64(ArcArray::from_shape_vec(IxDyn(&[]), vec![s]).unwrap()),
            DScalar::I8(x) => DTensor::I8(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
            DScalar::I16(x) => DTensor::I16(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
            DScalar::I32(x) => DTensor::I32(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
            DScalar::I64(x) => DTensor::I64(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
            DScalar::U8(x) => DTensor::U8(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
            DScalar::U16(x) => DTensor::U16(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
            DScalar::U32(x) => DTensor::U32(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
            DScalar::U64(x) => DTensor::U64(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
            DScalar::Bool(x) => DTensor::Bool(ArcArray::from_shape_vec(IxDyn(&[]), vec![x]).unwrap()),
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

    pub fn unwrap_int(self) -> Option<i128> {
        match self {
            DScalar::U8(x) => Some(x as i128),
            DScalar::U16(x) => Some(x as i128),
            DScalar::U32(x) => Some(x as i128),
            DScalar::U64(x) => Some(x as i128),
            DScalar::I8(x) => Some(x as i128),
            DScalar::I16(x) => Some(x as i128),
            DScalar::I32(x) => Some(x as i128),
            DScalar::I64(x) => Some(x as i128),
            _ => None,
        }
    }

    pub fn to_c_str(self) -> String {
        match self {
            DScalar::F32(c) => DisplayCFloat(*c as f64).to_string(),
            DScalar::F64(c) => DisplayCFloat(*c).to_string(),
            DScalar::U8(c) => format!("{}", c),
            DScalar::U16(c) => format!("{}", c),
            DScalar::U32(c) => format!("{}", c),
            DScalar::U64(c) => format!("{}", c),
            DScalar::I8(c) => format!("{}", c),
            DScalar::I16(c) => format!("{}", c),
            DScalar::I32(c) => format!("{}", c),
            DScalar::I64(c) => format!("{}", c),
            DScalar::Bool(c) => format!("{}", *c),
        }
    }

    pub fn value_cast(self, to: DType) -> DScalar {
        // cast to big general value
        let (yf, yi) = match self {
            DScalar::F32(T32(x)) => (x as f64, x as i128),
            DScalar::F64(T64(x)) => (x, x as i128),
            DScalar::I8(x) => (x as f64, x as i128),
            DScalar::I16(x) => (x as f64, x as i128),
            DScalar::I32(x) => (x as f64, x as i128),
            DScalar::I64(x) => (x as f64, x as i128),
            DScalar::U8(x) => (x as f64, x as i128),
            DScalar::U16(x) => (x as f64, x as i128),
            DScalar::U32(x) => (x as f64, x as i128),
            DScalar::U64(x) => (x as f64, x as i128),
            DScalar::Bool(DBool(x)) => (x as u8 as f64, x as u8 as i128),
        };

        // convert to target
        match to {
            DType::F32 => DScalar::f32(yf as f32),
            DType::F64 => DScalar::f64(yf),
            DType::I8 => DScalar::I8(yi as i8),
            DType::I16 => DScalar::I16(yi as i16),
            DType::I32 => DScalar::I32(yi as i32),
            DType::I64 => DScalar::I64(yi as i64),
            DType::U8 => DScalar::U8(yi as u8),
            DType::U16 => DScalar::U16(yi as u16),
            DType::U32 => DScalar::U32(yi as u32),
            DType::U64 => DScalar::U64(yi as u64),
            DType::Bool => DScalar::bool(yf != 0.0 || yi != 0),
        }
    }

    pub fn bit_cast(self, to: DType) -> Option<DScalar> {
        if self.dtype().size() != to.size() {
            return None;
        }

        // convert to bits, zero-extend just to be safe
        let bits = match self {
            DScalar::F32(T32(x)) => x.to_bits() as u64,
            DScalar::F64(T64(x)) => x.to_bits(),
            DScalar::I8(x) => x as u8 as u64,
            DScalar::I16(x) => x as u16 as u64,
            DScalar::I32(x) => x as u32 as u64,
            DScalar::I64(x) => x as u64,
            DScalar::U8(x) => x as u64,
            DScalar::U16(x) => x as u64,
            DScalar::U32(x) => x as u64,
            DScalar::U64(x) => x,
            DScalar::Bool(_) => return None,
        };

        // convert to target
        let y = match to {
            DType::F32 => DScalar::f32(f32::from_bits(bits as u32)),
            DType::F64 => DScalar::f64(f64::from_bits(bits)),
            DType::I8 => DScalar::I8(bits as i8),
            DType::I16 => DScalar::I16(bits as i16),
            DType::I32 => DScalar::I32(bits as i32),
            DType::I64 => DScalar::I64(bits as i64),
            DType::U8 => DScalar::U8(bits as u8),
            DType::U16 => DScalar::U16(bits as u16),
            DType::U32 => DScalar::U32(bits as u32),
            DType::U64 => DScalar::U64(bits),
            DType::Bool => return None,
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
impl_into_dscalar!(f64, DType::F64, F64, |x| DScalar::f64(x), DScalar::F64(T64(x)) => x);
impl_into_dscalar!(i8, DType::I8, I8, |x| DScalar::I8(x), DScalar::I8(x) => x);
impl_into_dscalar!(i16, DType::I16, I16, |x| DScalar::I16(x), DScalar::I16(x) => x);
impl_into_dscalar!(i32, DType::I32, I32, |x| DScalar::I32(x), DScalar::I32(x) => x);
impl_into_dscalar!(i64, DType::I64, I64, |x| DScalar::I64(x), DScalar::I64(x) => x);
impl_into_dscalar!(u8, DType::U8, U8, |x| DScalar::U8(x), DScalar::U8(x) => x);
impl_into_dscalar!(u16, DType::U16, U16, |x| DScalar::U16(x), DScalar::U16(x) => x);
impl_into_dscalar!(u32, DType::U32, U32, |x| DScalar::U32(x), DScalar::U32(x) => x);
impl_into_dscalar!(u64, DType::U64, U64, |x| DScalar::U64(x), DScalar::U64(x) => x);
impl_into_dscalar!(DBool, DType::Bool, Bool, |x| DScalar::Bool(x), DScalar::Bool(x) => x);

#[rustfmt::skip]
#[macro_export]
macro_rules! dispatch_dtensor {
    ($outer:expr, |$ty:ident, $f:ident, $inner:ident| $expr:expr) => {{
        use $crate::dtype::{DBool, DTensor};
        match $outer {
            DTensor::F32($inner) => { type $ty=f32; let $f=DTensor::F32; { $expr } }
            DTensor::F64($inner) => { type $ty=f64; let $f=DTensor::F64; { $expr } }
            DTensor::I8($inner) => { type $ty=i8; let $f=DTensor::I8; { $expr } }
            DTensor::I16($inner) => { type $ty=i16; let $f=DTensor::I16; { $expr } }
            DTensor::I32($inner) => { type $ty=i32; let $f=DTensor::I32; { $expr } }
            DTensor::I64($inner) => { type $ty=i64; let $f=DTensor::I64; { $expr } }
            DTensor::U8($inner) => { type $ty=u8; let $f=DTensor::U8; { $expr } }
            DTensor::U16($inner) => { type $ty=u16; let $f=DTensor::U16; { $expr } }
            DTensor::U32($inner) => { type $ty=u32; let $f=DTensor::U32; { $expr } }
            DTensor::U64($inner) => { type $ty=u64; let $f=DTensor::U64; { $expr } }
            DTensor::Bool($inner) => { type $ty=DBool; let $f=DTensor::Bool; { $expr } }
        }
    }};
}

#[rustfmt::skip]
#[macro_export]
macro_rules! dispatch_dtensor_pair {
    ($out_left:expr, $out_right:expr, |$ty:ident, $f:ident, $in_left:ident, $in_right:ident| $expr:expr) => {{
        use $crate::dtype::{DBool, DTensor};

        let out_left = $out_left;
        let out_right = $out_right;
        let dtype_left = out_left.dtype();
        let dtype_right = out_right.dtype();
        
        match (out_left, out_right) {
            (DTensor::F32($in_left), DTensor::F32($in_right)) => { type $ty=f32; let $f=DTensor::F32; { $expr } }
            (DTensor::I8($in_left), DTensor::I8($in_right)) => { type $ty=i8; let $f=DTensor::I8; { $expr } }
            (DTensor::I16($in_left), DTensor::I16($in_right)) => { type $ty=i16; let $f=DTensor::I16; { $expr } }
            (DTensor::I32($in_left), DTensor::I32($in_right)) => { type $ty=i32; let $f=DTensor::I32; { $expr } }
            (DTensor::I64($in_left), DTensor::I64($in_right)) => { type $ty=i64; let $f=DTensor::I64; { $expr } }
            (DTensor::U8($in_left), DTensor::U8($in_right)) => { type $ty=u8; let $f=DTensor::U8; { $expr } }
            (DTensor::U16($in_left), DTensor::U16($in_right)) => { type $ty=u16; let $f=DTensor::U16; { $expr } }
            (DTensor::U32($in_left), DTensor::U32($in_right)) => { type $ty=u32; let $f=DTensor::U32; { $expr } }
            (DTensor::U64($in_left), DTensor::U64($in_right)) => { type $ty=u64; let $f=DTensor::U64; { $expr } }
            (DTensor::Bool($in_left), DTensor::Bool($in_right)) => { type $ty=DBool; let $f=DTensor::Bool; { $expr } }
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
            (DScalar::Bool($in_left), DScalar::Bool($in_right)) => DScalar::Bool($expr),
            _ => panic!("Mismatched dtypes: left {:?}, right {:?}", out_left, out_right),
        }
    }}
}

// export macros
pub use dispatch_dtensor;
pub use dispatch_dtensor_pair;
pub use dispatch_dtype;
pub use map_dscalar_pair;
pub use map_dtensor;
pub use map_dtensor_pair;

impl DTensor {
    // TODO store shape and dtype in DTensor field so we don't have to pay this dispatch cost all the time?
    // TODO single accesor `shape_dtype`?
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

    pub fn unwrap_f64(&self) -> Option<&Tensor<f64>> {
        match self {
            DTensor::F64(tensor) => Some(tensor),
            _ => None,
        }
    }

    pub fn unwrap_i64(&self) -> Option<&Tensor<i64>> {
        match self {
            DTensor::I64(tensor) => Some(tensor),
            _ => None,
        }
    }

    pub fn unwrap_bool(&self) -> Option<&Tensor<DBool>> {
        match self {
            DTensor::Bool(tensor) => Some(tensor),
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
            (DTensor::F64(a), DTensor::F64(b)) => zip_eq(a.iter(), b.iter()).all(|(a, b)| a.float_eq(b)),

            // ints and bools can be compared like normal
            (DTensor::I8(a), DTensor::I8(b)) => a == b,
            (DTensor::I16(a), DTensor::I16(b)) => a == b,
            (DTensor::I32(a), DTensor::I32(b)) => a == b,
            (DTensor::I64(a), DTensor::I64(b)) => a == b,
            (DTensor::U8(a), DTensor::U8(b)) => a == b,
            (DTensor::U16(a), DTensor::U16(b)) => a == b,
            (DTensor::U32(a), DTensor::U32(b)) => a == b,
            (DTensor::U64(a), DTensor::U64(b)) => a == b,
            (DTensor::Bool(a), DTensor::Bool(b)) => a == b,

            // different types, not equal
            _ => false,
        }
    }
}

impl Hash for DTensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // hash shape, dtype and some of the first elements
        // (not all of them since that could be slow for large tensors)
        // TODO figure out how to include some of the middle and final elements too?
        const N: usize = 8;

        self.shape().hash(state);
        self.dtype().hash(state);

        match self {
            DTensor::F32(tensor) => tensor.iter().take(N).for_each(|x| x.float_hash(state)),
            DTensor::F64(tensor) => tensor.iter().take(N).for_each(|x| x.float_hash(state)),
            DTensor::I8(tensor) => tensor.iter().take(N).for_each(|x| x.hash(state)),
            DTensor::I16(tensor) => tensor.iter().take(N).for_each(|x| x.hash(state)),
            DTensor::I32(tensor) => tensor.iter().take(N).for_each(|x| x.hash(state)),
            DTensor::I64(tensor) => tensor.iter().take(N).for_each(|x| x.hash(state)),
            DTensor::U8(tensor) => tensor.iter().take(N).for_each(|x| x.hash(state)),
            DTensor::U16(tensor) => tensor.iter().take(N).for_each(|x| x.hash(state)),
            DTensor::U32(tensor) => tensor.iter().take(N).for_each(|x| x.hash(state)),
            DTensor::U64(tensor) => tensor.iter().take(N).for_each(|x| x.hash(state)),
            DTensor::Bool(tensor) => tensor.iter().take(N).for_each(|x| x.hash(state)),
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

impl Deref for T64 {
    type Target = f64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq<Self> for T64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.float_eq(&other.0)
    }
}

impl Eq for T64 {}

impl Hash for T64 {
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
pub struct DisplayCFloat(pub f64);

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

impl Deref for DBool {
    type Target = bool;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::Add for DBool {
    type Output = DBool;

    fn add(self, rhs: Self) -> Self::Output {
        DBool(self.0 || rhs.0)
    }
}

impl std::ops::Mul for DBool {
    type Output = DBool;

    fn mul(self, rhs: Self) -> Self::Output {
        DBool(self.0 && rhs.0)
    }
}

// sub and div don't make much sense
impl std::ops::Sub for DBool {
    type Output = DBool;

    fn sub(self, rhs: Self) -> Self::Output {
        DBool(self.0 && !rhs.0)
    }
}

impl std::ops::Div for DBool {
    type Output = DBool;

    fn div(self, rhs: Self) -> Self::Output {
        DBool(self.0 && !rhs.0)
    }
}

impl num_traits::Zero for DBool {
    fn zero() -> Self {
        DBool(false)
    }

    fn is_zero(&self) -> bool {
        !self.0
    }
}

impl num_traits::One for DBool {
    fn one() -> Self {
        DBool(true)
    }
}

unsafe impl NoUninit for DBool {}
