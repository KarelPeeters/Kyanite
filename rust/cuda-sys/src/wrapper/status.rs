use std::ffi::CStr;

use crate::bindings::{cudaError, cudaGetErrorString};
use crate::bindings::{cudnnGetErrorString, cudnnStatus_t};

pub trait Status {
    //TODO this is really static? or should we deallocate it?
    fn as_string(&self) -> &'static CStr;
    fn unwrap(self);
}

impl Status for cudaError {
    fn as_string(&self) -> &'static CStr {
        unsafe { CStr::from_ptr(cudaGetErrorString(*self)) }
    }

    fn unwrap(self) {
        if self != cudaError::cudaSuccess {
            panic!("Cuda operation returned error {:?}", self.as_string());
        }
    }
}

impl Status for cudnnStatus_t {
    fn as_string(&self) -> &'static CStr {
        unsafe { CStr::from_ptr(cudnnGetErrorString(*self)) }
    }

    fn unwrap(self) {
        if self != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            panic!("Cuda operation returned error {:?}", self.as_string());
        }
    }
}