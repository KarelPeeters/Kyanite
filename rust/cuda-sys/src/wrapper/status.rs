use std::ffi::CStr;

use crate::bindings::{cublasStatus_t, cudaError, cudaGetErrorString};
use crate::bindings::{cudnnGetErrorString, cudnnStatus_t};

pub trait Status {
    fn as_string(&self) -> &'static str;
    fn unwrap(&self);

    /// Alternative to `unwrap` that only panics if `!std::thread::panicking()`.
    /// This is useful to avoid double panics in [Drop] implementations.
    #[track_caller]
    fn unwrap_in_drop(&self) {
        if !std::thread::panicking() {
            self.unwrap();
        }
    }
}

impl Status for cudaError {
    fn as_string(&self) -> &'static str {
        unsafe { CStr::from_ptr(cudaGetErrorString(*self)) }.to_str().unwrap()
    }

    #[track_caller]
    fn unwrap(&self) {
        if *self != cudaError::cudaSuccess {
            panic!("Cuda operation returned error {:?}", self.as_string());
        }
    }
}

impl Status for cudnnStatus_t {
    fn as_string(&self) -> &'static str {
        unsafe { CStr::from_ptr(cudnnGetErrorString(*self)) }.to_str().unwrap()
    }

    #[track_caller]
    fn unwrap(&self) {
        if *self != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            panic!("Cudnn operation returned error {:?}", self);
        }
    }
}

impl Status for cublasStatus_t {
    fn as_string(&self) -> &'static str {
        match self {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => "CUBLAS_STATUS_SUCCESS",
            cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => "CUBLAS_STATUS_NOT_INITIALIZED",
            cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED => "CUBLAS_STATUS_ALLOC_FAILED",
            cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => "CUBLAS_STATUS_INVALID_VALUE",
            cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => "CUBLAS_STATUS_ARCH_MISMATCH",
            cublasStatus_t::CUBLAS_STATUS_MAPPING_ERROR => "CUBLAS_STATUS_MAPPING_ERROR",
            cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => "CUBLAS_STATUS_EXECUTION_FAILED",
            cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR => "CUBLAS_STATUS_INTERNAL_ERROR",
            cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED => "CUBLAS_STATUS_NOT_SUPPORTED",
            cublasStatus_t::CUBLAS_STATUS_LICENSE_ERROR => "CUBLAS_STATUS_LICENSE_ERROR",
        }
    }

    #[track_caller]
    fn unwrap(&self) {
        if *self != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            panic!("Cublas operation returned error {:?}", self.as_string());
        }
    }
}
