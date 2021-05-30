use std::fmt::{Debug, Formatter, Display};
use std::ops::{Deref, DerefMut};

/// A wrapper around `f32` that implements `Eq`.
#[derive(Copy, Clone)]
pub struct EqF32(pub f32);

impl Deref for EqF32 {
    type Target = f32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for EqF32 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<f32> for EqF32 {
    fn from(f: f32) -> Self {
        EqF32(f)
    }
}

impl Into<f32> for EqF32 {
    fn into(self) -> f32 {
        self.0
    }
}

impl PartialEq for EqF32 {
    fn eq(&self, other: &Self) -> bool {
        (self.0.is_nan() && other.0.is_nan()) || (self.0 == other.0)
    }
}

impl Eq for EqF32 {}

impl Debug for EqF32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Display for EqF32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}