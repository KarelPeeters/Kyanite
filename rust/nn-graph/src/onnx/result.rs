use std::io;
use std::path::{Path, PathBuf};

pub type OnnxResult<T> = Result<T, OnnxLoadError>;

#[derive(Debug)]
pub enum OnnxLoadError {
    IO(PathBuf, io::Error),
    NonNormalExternalDataPath(PathBuf),
}

pub trait ToOnnxLoadResult {
    type T;
    fn to_onnx_result(self, path: impl AsRef<Path>) -> OnnxResult<Self::T>;
}

impl<T> ToOnnxLoadResult for Result<T, io::Error> {
    type T = T;
    fn to_onnx_result(self, path: impl AsRef<Path>) -> OnnxResult<T> {
        self.map_err(|e| OnnxLoadError::IO(path.as_ref().to_owned(), e))
    }
}
