use std::fmt::{Debug, Formatter};
use std::ops::{Deref, DerefMut};

/// A newtype that implements debug by only printing the name of the contained type.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct WrapDebug<T>(T);

impl<T> WrapDebug<T> {
    pub fn inner(&self) -> &T {
        &self.0
    }
}

impl<T> Debug for WrapDebug<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", std::any::type_name::<T>())
    }
}

impl<T> From<T> for WrapDebug<T> {
    fn from(value: T) -> Self {
        WrapDebug(value)
    }
}

impl<T> Deref for WrapDebug<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for WrapDebug<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
