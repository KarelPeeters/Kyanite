use std::ptr::null_mut;

use crate::bindings::{cudaGetDeviceCount, cudaSetDevice, cudnnCreate, cudnnDestroy};
use crate::bindings::cudnnHandle_t;
use crate::wrapper::status::Status;

pub fn cuda_device_count() -> i32 {
    unsafe {
        let mut count = 0;
        cudaGetDeviceCount(&mut count as *mut _).unwrap();
        count
    }
}

pub unsafe fn cuda_set_device(device: i32) {
    cudaSetDevice(device).unwrap()
}

pub struct CudnnHandle {
    inner: cudnnHandle_t,
    device: i32,
}

impl Drop for CudnnHandle {
    fn drop(&mut self) {
        unsafe {
            cuda_set_device(self.device);
            cudnnDestroy(self.inner).unwrap()
        }
    }
}

impl CudnnHandle {
    pub fn new(device: i32) -> Self {
        unsafe {
            let mut inner = null_mut();
            cuda_set_device(device);
            cudnnCreate(&mut inner as *mut _).unwrap();
            CudnnHandle { inner, device }
        }
    }

    pub unsafe fn switch_to_device(self) {
        cuda_set_device(self.device);
    }

    pub fn device(&self)  -> i32 {
        self.device
    }

    pub unsafe fn inner(&mut self) -> cudnnHandle_t {
        self.inner
    }
}
