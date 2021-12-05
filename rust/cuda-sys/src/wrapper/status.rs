use std::ffi::CStr;

use crate::bindings::{cudaError, cudaGetErrorString};
use crate::bindings::{cudnnGetErrorString, cudnnStatus_t};

pub trait Status {
    //TODO this is really static? or should we deallocate it?
    fn as_string(&self) -> &'static CStr;
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
    fn as_string(&self) -> &'static CStr {
        unsafe { CStr::from_ptr(cudaGetErrorString(*self)) }
    }

    #[track_caller]
    fn unwrap(&self) {
        if *self != cudaError::cudaSuccess {
            panic!("Cuda operation returned error {:?}", self.as_string());
        }
    }
}

impl Status for cudnnStatus_t {
    fn as_string(&self) -> &'static CStr {
        unsafe { CStr::from_ptr(cudnnGetErrorString(*self)) }
    }

    #[track_caller]
    fn unwrap(&self) {
        if *self != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            panic!("Cuda operation returned error {:?}", self.as_string());
        }
    }
}