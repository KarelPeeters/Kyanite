use std::ffi::CStr;
use std::ptr::null;

use crate::bindings::{
    cuGetErrorName, cublasStatus_t, cudaError, cudaGetErrorString, nvrtcGetErrorString, nvrtcResult, CUresult,
};
use crate::bindings::{cudnnGetErrorString, cudnnStatus_t};

pub trait Status: Copy + PartialEq {
    const SUCCESS: Self;

    fn as_string(&self) -> &'static str;

    fn is_success(&self) -> bool {
        *self == Self::SUCCESS
    }

    #[track_caller]
    fn unwrap(&self) {
        if !self.is_success() {
            panic!("Operation returned error {:?}", self.as_string());
        }
    }

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
    const SUCCESS: Self = cudaError::cudaSuccess;

    fn as_string(&self) -> &'static str {
        unsafe { CStr::from_ptr(cudaGetErrorString(*self)) }.to_str().unwrap()
    }
}

impl Status for cudnnStatus_t {
    const SUCCESS: Self = cudnnStatus_t::CUDNN_STATUS_SUCCESS;

    fn as_string(&self) -> &'static str {
        unsafe { CStr::from_ptr(cudnnGetErrorString(*self)) }.to_str().unwrap()
    }
}

impl Status for cublasStatus_t {
    const SUCCESS: Self = cublasStatus_t::CUBLAS_STATUS_SUCCESS;

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
}

impl Status for nvrtcResult {
    const SUCCESS: Self = nvrtcResult::NVRTC_SUCCESS;

    fn as_string(&self) -> &'static str {
        unsafe { CStr::from_ptr(nvrtcGetErrorString(*self)) }.to_str().unwrap()
    }
}

impl Status for CUresult {
    const SUCCESS: Self = CUresult::CUDA_SUCCESS;

    fn as_string(&self) -> &'static str {
        unsafe {
            let mut ptr = null();
            let result = cuGetErrorName(*self, &mut ptr as *mut _);
            if result != CUresult::CUDA_SUCCESS {
                panic!("Error '{:?}' while getting name of error '{:?}'", result, self);
            }
            CStr::from_ptr(ptr).to_str().unwrap()
        }
    }
}
