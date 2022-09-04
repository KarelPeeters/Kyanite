use std::fmt::Formatter;
use std::marker::PhantomData;
use std::ops::Deref;
use std::str::FromStr;

use serde::de::{Unexpected, Visitor};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Copy, Clone)]
pub struct ToFromStringArg<T>(pub T);

impl<T> Deref for ToFromStringArg<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: ToString> Serialize for ToFromStringArg<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0.to_string())
    }
}

impl<'de, T: FromStr> Deserialize<'de> for ToFromStringArg<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(ToFromStringArgVisitor(PhantomData))
    }
}

struct ToFromStringArgVisitor<T>(PhantomData<T>);

impl<'de, T: FromStr> Visitor<'de> for ToFromStringArgVisitor<T> {
    type Value = ToFromStringArg<T>;

    fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
        write!(formatter, "an fpu mode")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        T::from_str(v)
            .map(ToFromStringArg)
            .map_err(|_| E::invalid_value(Unexpected::Str(v), &self))
    }
}
