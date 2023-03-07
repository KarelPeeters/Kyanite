use std::io;
use std::path::{Path, PathBuf};

pub type OnnxResult<T> = Result<T, OnnxError>;

#[derive(Debug)]
pub enum OnnxError {
    IO(PathBuf, io::Error),

    NonNormalExternalDataPath(PathBuf),
    MustHaveParentPath(PathBuf),

    MissingProtoField(&'static str),

    LeftoverInputs(String, Vec<usize>),
    LeftoverAttributes(String, Vec<String>),

    InvalidOperationArgs { node: String, message: String },

    UnsupportedOperation(String, String),
    UnsupportedNonFloatOutput(String),
    UnsupportedNdConvolution(String, usize),
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
