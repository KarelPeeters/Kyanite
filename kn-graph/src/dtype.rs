use decorum::Total;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum DType {
    F32,
    I(DSize),
    U(DSize),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum DSize {
    S8,
    S16,
    S32,
    S64,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DConst {
    F32(Total<f32>),
    I(DSize, i64),
    U(DSize, u64),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Specials {
    pub min: DConst,
    pub max: DConst,
    pub zero: DConst,
    pub one: DConst,
}

impl DType {
    pub fn size(self) -> DSize {
        match self {
            DType::F32 => DSize::S32,
            DType::I(size) => size,
            DType::U(size) => size,
        }
    }

    pub fn is_signed(self) -> bool {
        match self {
            DType::F32 => true,
            DType::I(_) => true,
            DType::U(_) => false,
        }
    }

    pub fn is_float(self) -> bool {
        match self {
            DType::F32 => true,
            DType::I(_) => false,
            DType::U(_) => false,
        }
    }

    pub fn specials(self) -> Specials {
        // TODO compact this
        match self {
            DType::F32 => Specials {
                min: DConst::f32(f32::NEG_INFINITY),
                max: DConst::f32(f32::INFINITY),
                zero: DConst::f32(0.0),
                one: DConst::f32(1.0),
            },
            DType::I(size) => {
                let (min, max) = match size {
                    DSize::S8 => (i8::MIN as i64, i8::MAX as i64),
                    DSize::S16 => (i16::MIN as i64, i16::MAX as i64),
                    DSize::S32 => (i32::MIN as i64, i32::MAX as i64),
                    DSize::S64 => (i64::MIN, i64::MAX),
                };
                Specials {
                    min: DConst::I(size, min),
                    max: DConst::I(size, max),
                    zero: DConst::I(size, 0),
                    one: DConst::I(size, 1),
                }
            }
            DType::U(size) => {
                let (min, max) = match size {
                    DSize::S8 => (u8::MIN as u64, u8::MAX as u64),
                    DSize::S16 => (u16::MIN as u64, u16::MAX as u64),
                    DSize::S32 => (u32::MIN as u64, u32::MAX as u64),
                    DSize::S64 => (u64::MIN, u64::MAX),
                };
                Specials {
                    min: DConst::U(size, min),
                    max: DConst::U(size, max),
                    zero: DConst::U(size, 0),
                    one: DConst::U(size, 1),
                }
            }
        }
    }

    pub fn iter_bytes(self, bytes: &[u8]) -> Option<impl Iterator<Item = DConst> + '_> {
        let size = self.size().bytes();
        if bytes.len() % size != 0 {
            return None;
        }

        // TODO compact this
        Some(bytes.chunks_exact(size).map(move |x| match self {
            DType::F32 => DConst::f32(f32::from_le_bytes(x.try_into().unwrap())),
            DType::I(DSize::S8) => DConst::I(DSize::S8, i8::from_le_bytes(x.try_into().unwrap()) as i64),
            DType::I(DSize::S16) => DConst::I(DSize::S16, i16::from_le_bytes(x.try_into().unwrap()) as i64),
            DType::I(DSize::S32) => DConst::I(DSize::S32, i32::from_le_bytes(x.try_into().unwrap()) as i64),
            DType::I(DSize::S64) => DConst::I(DSize::S64, i64::from_le_bytes(x.try_into().unwrap())),
            DType::U(DSize::S8) => DConst::U(DSize::S8, u8::from_le_bytes(x.try_into().unwrap()) as u64),
            DType::U(DSize::S16) => DConst::U(DSize::S16, u16::from_le_bytes(x.try_into().unwrap()) as u64),
            DType::U(DSize::S32) => DConst::U(DSize::S32, u32::from_le_bytes(x.try_into().unwrap()) as u64),
            DType::U(DSize::S64) => DConst::U(DSize::S64, u64::from_le_bytes(x.try_into().unwrap())),
        }))
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

impl DConst {
    pub fn f32(x: f32) -> Self {
        DConst::F32(Total::from_inner(x))
    }

    pub fn dtype(self) -> DType {
        match self {
            DConst::F32(_) => DType::F32,
            DConst::I(size, _) => DType::I(size),
            DConst::U(size, _) => DType::U(size),
        }
    }

    pub fn to_bytes(self) -> Vec<u8> {
        match self {
            DConst::F32(x) => x.into_inner().to_le_bytes().to_vec(),
            DConst::I(size, x) => x.to_le_bytes()[..size.bytes()].to_vec(),
            DConst::U(size, x) => x.to_le_bytes()[..size.bytes()].to_vec(),
        }
    }

    pub fn unwrap_f32(self) -> Option<f32> {
        match self {
            DConst::F32(x) => Some(x.into_inner()),
            _ => None,
        }
    }
}

pub trait IntoDConst: Copy {
    const DTYPE: DType;
    fn to_dconst(&self) -> DConst;
}

macro_rules! impl_into_dconst {
    ($ty:ty, $dtype:expr, |$x:ident| $conv:expr) => {
        impl IntoDConst for $ty {
            const DTYPE: DType = $dtype;

            fn to_dconst(&self) -> DConst {
                let &$x = self;
                $conv
            }
        }
    };
}

impl_into_dconst!(f32, DType::F32, |x| DConst::f32(x));
impl_into_dconst!(i8, DType::I(DSize::S8), |x| DConst::I(DSize::S8, x as i64));
impl_into_dconst!(i16, DType::I(DSize::S16), |x| DConst::I(DSize::S16, x as i64));
impl_into_dconst!(i32, DType::I(DSize::S32), |x| DConst::I(DSize::S32, x as i64));
impl_into_dconst!(i64, DType::I(DSize::S64), |x| DConst::I(DSize::S64, x));
impl_into_dconst!(u8, DType::U(DSize::S8), |x| DConst::U(DSize::S8, x as u64));
impl_into_dconst!(u16, DType::U(DSize::S16), |x| DConst::U(DSize::S16, x as u64));
impl_into_dconst!(u32, DType::U(DSize::S32), |x| DConst::U(DSize::S32, x as u64));
impl_into_dconst!(u64, DType::U(DSize::S64), |x| DConst::U(DSize::S64, x));
