use std::ptr::null_mut;

use crate::bindings::{cudaGetDevice, cudaGetDeviceCount, cudaSetDevice, cudnnCreate, cudnnDestroy};
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

pub fn cuda_device_count() -> i32 {
    unsafe {
        let mut count = 0;
        cudaGetDeviceCount(&mut count as *mut _).unwrap();
        count
    }
}

impl CudnnHandle {
    pub fn new(device: i32) -> Self {
        unsafe {
            // keep the current device so we can restore it later
            let mut prev = 0;
            cudaGetDevice(&mut prev as *mut _).unwrap();

            // set the requested device and create the cudnn handle
            cudaSetDevice(device).unwrap();
            let mut inner = null_mut();
            cudnnCreate(&mut inner as *mut _).unwrap();

            // restore previous device
            cudaSetDevice(prev).unwrap();

            CudnnHandle(inner)
        }
    }

    pub unsafe fn inner(&mut self) -> cudnnHandle_t {
        self.0
    }
}
