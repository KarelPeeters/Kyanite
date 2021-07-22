use std::ptr::null_mut;

use crate::bindings::{cudnnCreate, cudnnDestroy};
use crate::bindings::cudnnHandle_t;
use crate::wrapper::status::Status;

pub struct CudnnHandle(cudnnHandle_t);

impl Drop for CudnnHandle {
    fn drop(&mut self) {
        unsafe {
            cudnnDestroy(self.0).unwrap()
        }
    }
}

impl CudnnHandle {
    pub fn new() -> Self {
        unsafe {
            let mut inner = null_mut();
            cudnnCreate(&mut inner as *mut _).unwrap();
            CudnnHandle(inner)
        }
    }

    pub unsafe fn inner(&mut self) -> cudnnHandle_t {
        self.0
    }
}
