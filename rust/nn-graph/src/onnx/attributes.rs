use std::collections::HashMap;

use itertools::Itertools;

use crate::onnx::proto::{AttributeProto, TensorProto};
use crate::onnx::proto::attribute_proto::AttributeType;

#[derive(Debug)]
pub struct Attributes<'a> {
    inner: HashMap<&'a str, &'a AttributeProto>,
}

impl<'a> Attributes<'a> {
    pub fn from(attrs: &'a [AttributeProto]) -> Self {
        let inner: HashMap<&str, &AttributeProto> = attrs
            .iter()
            .map(|a| (&*a.name, a))
            .collect();
        Attributes { inner }
    }

    pub fn take(&mut self, key: &str, ty: AttributeType) -> &'a AttributeProto {
        let attribute = self.inner.remove(key)
            .unwrap_or_else(|| {
                let available = self.inner.keys().collect_vec();
                panic!("Missing attribute {}, available: {:?}", key, available)
            });
        assert_eq!(ty, attribute.r#type(), "Expected type {:?}", ty);
        attribute
    }

    pub fn take_int(&mut self, key: &str) -> i64 {
        self.take(key, AttributeType::Int).i
    }

    pub fn take_ints(&mut self, key: &str) -> &'a [i64] {
        &self.take(key, AttributeType::Ints).ints
    }

    pub fn take_float(&mut self, key: &str) -> f32 {
        self.take(key, AttributeType::Float).f
    }

    pub fn take_tensor(&mut self, key: &str) -> &TensorProto {
        self.take(key, AttributeType::Tensor).t.as_ref().unwrap()
    }

    pub fn is_done(&self) -> bool {
        self.inner.is_empty()
    }
}
