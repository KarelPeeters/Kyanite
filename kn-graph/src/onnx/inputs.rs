use std::collections::HashMap;

use itertools::Itertools;

use crate::onnx::proto::attribute_proto::AttributeType;
use crate::onnx::proto::{AttributeProto, TensorProto};
use crate::onnx::result::{Node, OnnxError, OnnxResult};
use crate::onnx::store::Store;
use crate::onnx::typed_value::TypedValue;

#[derive(Debug)]
pub struct Inputs<'a> {
    node: Node<&'a str>,
    inner: Vec<Storage<&'a TypedValue>>,
}

#[derive(Debug)]
enum Storage<T> {
    Missing,
    Used,
    Present(T),
}

impl<'a> Inputs<'a> {
    pub fn from(node: Node<&'a str>, inputs: &'a [String], nodes: &'a Store<TypedValue>) -> OnnxResult<Self> {
        let inner = inputs
            .iter()
            .enumerate()
            .map(|(i, name)| {
                // an empty attribute name means the input is missing (which is only allowed for optional inputs)
                if name.is_empty() {
                    Ok(Storage::Missing)
                } else {
                    let value = nodes
                        .get(name)
                        .ok_or_else(|| OnnxError::InputNodeDoesNotExist(node.to_owned(), i, name.to_owned()))?;
                    Ok(Storage::Present(value))
                }
            })
            .try_collect()?;

        Ok(Inputs { node, inner })
    }

    pub fn required(&mut self, index: usize) -> OnnxResult<&'a TypedValue> {
        match self.optional(index) {
            Some(input) => Ok(input),
            None => Err(OnnxError::MissingInput(self.node.to_owned(), index, self.inner.len())),
        }
    }

    pub fn optional(&mut self, index: usize) -> Option<&'a TypedValue> {
        match self.take(index) {
            Storage::Present(value) => Some(value),
            Storage::Missing => None,
            Storage::Used => {
                // this is a panic since this is always a bug in the code, not an invalid input
                panic!("Already used input {} of node {:?}", index, self.node)
            }
        }
    }

    pub fn take_all_variadic(&mut self) -> Vec<&'a TypedValue> {
        (0..self.inner.len())
            .map(|i| match self.take(i) {
                Storage::Present(value) => value,
                // TODO potentially replace these panics with errors (although not if they can only be caused by bugs)
                Storage::Used => panic!(
                    "Cannot get variadic input, input {} of node {:?} has already been used",
                    i, self.node
                ),
                Storage::Missing => panic!("Missing input {} not allowed in variadic on node {:?}", i, self.node),
            })
            .collect()
    }

    pub fn leftover(&mut self) -> Vec<usize> {
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
    node: Node<&'a str>,
    inner: HashMap<&'a str, &'a AttributeProto>,
}

#[allow(dead_code)]
impl<'a> Attributes<'a> {
    pub fn from(node: Node<&'a str>, attrs: &'a [AttributeProto]) -> Self {
        let inner: HashMap<&str, &AttributeProto> = attrs.iter().map(|a| (&*a.name, a)).collect();
        Attributes { inner, node }
    }

    pub fn maybe_take(&mut self, key: &str, ty: AttributeType) -> OnnxResult<Option<&'a AttributeProto>> {
        self.inner
            .remove(key)
            .map(|attribute| {
                let actual = attribute.r#type();
                if ty == actual {
                    Ok(attribute)
                } else {
                    Err(OnnxError::UnexpectedAttributeType(
                        self.node.to_owned(),
                        key.to_owned(),
                        ty,
                        actual,
                    ))
                }
            })
            .transpose()
    }

    pub fn take(&mut self, key: &str, ty: AttributeType) -> OnnxResult<&'a AttributeProto> {
        self.maybe_take(key, ty)?.ok_or_else(|| {
            let available = self.inner.keys().map(|&s| s.to_owned()).collect_vec();
            OnnxError::MissingAttribute(self.node.to_owned(), key.to_owned(), ty, available)
        })
    }

    pub fn maybe_take_string(&mut self, key: &str) -> OnnxResult<Option<&str>> {
        Ok(self
            .maybe_take(key, AttributeType::String)?
            .map(|s| std::str::from_utf8(&s.s).unwrap()))
    }

    pub fn take_string(&mut self, key: &str) -> OnnxResult<&str> {
        Ok(std::str::from_utf8(&self.take(key, AttributeType::String)?.s).unwrap())
    }

    pub fn maybe_take_int(&mut self, key: &str) -> OnnxResult<Option<i64>> {
        Ok(self.maybe_take(key, AttributeType::Int)?.map(|a| a.i))
    }

    pub fn take_int(&mut self, key: &str) -> OnnxResult<i64> {
        Ok(self.take(key, AttributeType::Int)?.i)
    }

    pub fn maybe_take_bool(&mut self, key: &str) -> OnnxResult<Option<bool>> {
        match self.maybe_take(key, AttributeType::Int)? {
            None => Ok(None),
            Some(a) => Ok(Some(map_bool(self.node, key, a.i)?)),
        }
    }

    pub fn take_bool(&mut self, key: &str) -> OnnxResult<bool> {
        let i = self.take(key, AttributeType::Int)?.i;
        map_bool(self.node, key, i)
    }

    pub fn maybe_take_ints(&mut self, key: &str) -> OnnxResult<Option<&'a [i64]>> {
        Ok(self.maybe_take(key, AttributeType::Ints)?.map(|a| &*a.ints))
    }

    pub fn take_ints(&mut self, key: &str) -> OnnxResult<&'a [i64]> {
        Ok(&self.take(key, AttributeType::Ints)?.ints)
    }

    pub fn maybe_take_float(&mut self, key: &str) -> OnnxResult<Option<f32>> {
        Ok(self.maybe_take(key, AttributeType::Float)?.map(|a| a.f))
    }

    pub fn take_float(&mut self, key: &str) -> OnnxResult<f32> {
        Ok(self.take(key, AttributeType::Float)?.f)
    }

    pub fn maybe_take_tensor(&mut self, key: &str) -> OnnxResult<Option<&TensorProto>> {
        Ok(self
            .maybe_take(key, AttributeType::Tensor)?
            .map(|a| a.t.as_ref().unwrap()))
    }

    pub fn take_tensor(&mut self, key: &str) -> OnnxResult<&TensorProto> {
        Ok(self.take(key, AttributeType::Tensor)?.t.as_ref().unwrap())
    }

    pub fn leftover(&self) -> Vec<String> {
        self.inner.keys().map(|&s| s.to_owned()).collect()
    }
}

fn map_bool(node: Node<&str>, key: &str, i: i64) -> OnnxResult<bool> {
    if i == 0 || i == 1 {
        Ok(i != 0)
    } else {
        Err(OnnxError::InvalidAttributeBool(node.to_owned(), key.to_owned(), i))
    }
}
