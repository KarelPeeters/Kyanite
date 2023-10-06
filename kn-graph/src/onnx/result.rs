use std::error::Error;
use std::fmt::{Display, Formatter};
use std::io;
use std::path::{Path, PathBuf};

use crate::onnx::proto::attribute_proto::AttributeType;
use crate::onnx::proto::tensor_proto::DataType;
use crate::onnx::typed_value::AsShapeError;

pub type OnnxResult<T> = Result<T, OnnxError>;

#[derive(Debug, Copy, Clone)]
pub struct Node<S = String> {
    pub name: S,
    pub op_type: S,
}

#[derive(Debug)]
pub enum OnnxError {
    IO(PathBuf, io::Error),

    NonNormalExternalDataPath(PathBuf),
    MustHaveParentPath(PathBuf),

    MissingProtoField(&'static str),

    LeftoverInputs(Node, Vec<usize>),
    LeftoverAttributes(Node, Vec<String>),

    InvalidOperationArgs(Node, String),
    InputNodeDoesNotExist(Node, usize, String),
    MissingInput(Node, usize, usize),
    MissingAttribute(Node, String, AttributeType, Vec<String>),
    UnexpectedAttributeType(Node, String, AttributeType, AttributeType),
    InvalidAttributeBool(Node, String, i64),

    UnsupportedOperation(Node),

    UnsupportedMultipleOutputs(Node, Vec<String>),
    UnsupportedNonFloatOutput(String),
    UnsupportedType(String, DataType),

    UnsupportedNdConvolution(Node, usize),

    UnsupportedPartialShape(Node, String),
    UnsupportedShape(Node, String),

    UnsupportedElementWiseCombination(Node, String, String),

    //TODO node/operand info
    ExpectedNonBatchValue(String),
    ExpectedSizeError(AsShapeError),
}

impl From<AsShapeError> for OnnxError {
    fn from(e: AsShapeError) -> Self {
        OnnxError::ExpectedSizeError(e)
    }
}

pub trait ToOnnxLoadResult {
    type T;
    fn to_onnx_result(self, path: impl AsRef<Path>) -> OnnxResult<Self::T>;
}

impl<T> ToOnnxLoadResult for Result<T, io::Error> {
    type T = T;
    fn to_onnx_result(self, path: impl AsRef<Path>) -> OnnxResult<T> {
        self.map_err(|e| OnnxError::IO(path.as_ref().to_owned(), e))
    }
}

pub trait UnwrapProto {
    type T;
    fn unwrap_proto(self, field: &'static str) -> OnnxResult<Self::T>;
}

impl<T> UnwrapProto for Option<T> {
    type T = T;
    fn unwrap_proto(self, field: &'static str) -> OnnxResult<T> {
        self.ok_or(OnnxError::MissingProtoField(field))
    }
}

impl<S: AsRef<str>> Node<S> {
    pub fn to_owned(self) -> Node<String> {
        Node {
            name: self.name.as_ref().to_owned(),
            op_type: self.op_type.as_ref().to_owned(),
        }
    }
}

impl Display for OnnxError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for OnnxError {}
