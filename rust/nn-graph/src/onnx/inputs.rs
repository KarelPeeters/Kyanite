use std::collections::HashMap;

use itertools::Itertools;

use crate::onnx::proto::attribute_proto::AttributeType;
use crate::onnx::proto::{AttributeProto, TensorProto};
use crate::onnx::store::Store;
use crate::onnx::typed_value::TypedValue;

#[derive(Debug)]
pub struct Inputs<'a> {
    node_name: String,
    inner: Vec<Storage<&'a TypedValue>>,
}

#[derive(Debug)]
enum Storage<T> {
    Missing,
    Used,
    Present(T),
}

impl<'a> Inputs<'a> {
    pub fn from(node_name: String, inputs: &'a Vec<String>, nodes: &'a Store<TypedValue>) -> Self {
        let inner = inputs
            .iter()
            .enumerate()
            .map(|(i, name)| {
                // an empty attribute name means the input is missing (which is only allowed for optional inputs)
                if name == "" {
                    Storage::Missing
                } else {
                    let value = nodes
                        .get(name)
                        .unwrap_or_else(|| panic!("Input {} {} of node {} not found", i, name, node_name));
                    Storage::Present(value)
                }
            })
            .collect_vec();

        Inputs { inner, node_name }
    }

    pub fn required(&mut self, index: usize) -> &'a TypedValue {
        match self.optional(index) {
            Some(input) => input,
            None => panic!(
                "Missing input {} of node {}, {} inputs were provided",
                index,
                self.node_name,
                self.inner.len()
            ),
        }
    }

    pub fn optional(&mut self, index: usize) -> Option<&'a TypedValue> {
        match self.take(index) {
            Storage::Present(value) => Some(value),
            Storage::Missing => None,
            Storage::Used => {
                panic!("Already used input {} of node {}", index, self.node_name)
            }
        }
    }

    pub fn take_all_varadic(&mut self) -> Vec<&'a TypedValue> {
        (0..self.inner.len())
            .map(|i| match self.take(i) {
                Storage::Present(value) => value,
                Storage::Used => panic!(
                    "Cannot get varadic input, input {} of node {} has already been used",
                    i, self.node_name
                ),
                Storage::Missing => panic!("Missing input {} not allowed in varadic on node {}", i, self.node_name),
            })
            .collect()
    }

    pub fn leftover_inputs(&mut self) -> Vec<usize> {
        self.inner
            .iter()
            .positions(|x| matches!(x, Storage::Present(_)))
            .collect()
    }

    fn take(&mut self, index: usize) -> Storage<&'a TypedValue> {
        match self.inner.get(index) {
            None => Storage::Missing,
            Some(Storage::Missing) => Storage::Missing,
            Some(Storage::Used) => Storage::Used,
            Some(&Storage::Present(value)) => {
                self.inner[index] = Storage::Used;
                Storage::Present(value)
            }
        }
    }
}

#[derive(Debug)]
pub struct Attributes<'a> {
    inner: HashMap<&'a str, &'a AttributeProto>,
}

#[allow(dead_code)]
impl<'a> Attributes<'a> {
    pub fn from(attrs: &'a [AttributeProto]) -> Self {
        let inner: HashMap<&str, &AttributeProto> = attrs.iter().map(|a| (&*a.name, a)).collect();
        Attributes { inner }
    }

    pub fn maybe_take(&mut self, key: &str, ty: AttributeType) -> Option<&'a AttributeProto> {
        self.inner.remove(key).map(|attribute| {
            assert_eq!(ty, attribute.r#type(), "Expected type {:?}", ty);
            attribute
        })
    }

    pub fn take(&mut self, key: &str, ty: AttributeType) -> &'a AttributeProto {
        self.maybe_take(key, ty).unwrap_or_else(|| {
            let available = self.inner.keys().collect_vec();
            panic!("Missing attribute {}, available: {:?}", key, available)
        })
    }

    pub fn maybe_take_string(&mut self, key: &str) -> Option<&str> {
        self.maybe_take(key, AttributeType::String)
            .map(|s| std::str::from_utf8(&s.s).unwrap())
    }

    pub fn take_string(&mut self, key: &str) -> &str {
        std::str::from_utf8(&self.take(key, AttributeType::String).s).unwrap()
    }

    pub fn maybe_take_int(&mut self, key: &str) -> Option<i64> {
        self.maybe_take(key, AttributeType::Int).map(|a| a.i)
    }

    pub fn take_int(&mut self, key: &str) -> i64 {
        self.take(key, AttributeType::Int).i
    }

    pub fn maybe_take_bool(&mut self, key: &str) -> Option<bool> {
        self.maybe_take(key, AttributeType::Int).map(|a| map_bool(key, a.i))
    }

    pub fn take_bool(&mut self, key: &str) -> bool {
        map_bool(key, self.take(key, AttributeType::Int).i)
    }

    pub fn maybe_take_ints(&mut self, key: &str) -> Option<&'a [i64]> {
        self.maybe_take(key, AttributeType::Ints).map(|a| &*a.ints)
    }

    pub fn take_ints(&mut self, key: &str) -> &'a [i64] {
        &self.take(key, AttributeType::Ints).ints
    }

    pub fn maybe_take_float(&mut self, key: &str) -> Option<f32> {
        self.maybe_take(key, AttributeType::Float).map(|a| a.f)
    }

    pub fn take_float(&mut self, key: &str) -> f32 {
        self.take(key, AttributeType::Float).f
    }

    pub fn maybe_take_tensor(&mut self, key: &str) -> Option<&TensorProto> {
        self.maybe_take(key, AttributeType::Tensor)
            .map(|a| a.t.as_ref().unwrap())
    }

    pub fn take_tensor(&mut self, key: &str) -> &TensorProto {
        self.take(key, AttributeType::Tensor).t.as_ref().unwrap()
    }

    pub fn is_done(&self) -> bool {
        self.inner.is_empty()
    }
}

fn map_bool(key: &str, i: i64) -> bool {
    assert!(
        i == 0 || i == 1,
        "Attribute {} is a bool and should be 0 or 1, got {}",
        key,
        i
    );
    i != 0
}
