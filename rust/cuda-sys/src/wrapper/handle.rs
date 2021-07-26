use std::ptr::null_mut;

use crate::bindings::{cudaGetDeviceCount, cudaSetDevice, cudaStream_t, cudaStreamCreate, cudaStreamDestroy, cudnnCreate, cudnnDestroy, cudnnSetStream};
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

//TODO copy? clone? default stream?
#[derive(Debug)]
pub struct CudaStream {
    device: i32,
    inner: cudaStream_t,
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            cudaStreamDestroy(self.inner).unwrap();
        }
    }
}

impl CudaStream {
    pub fn new(device: i32) -> Self {
        unsafe {
            let mut inner = null_mut();
            cuda_set_device(device);
            cudaStreamCreate(&mut inner as *mut _).unwrap();
            CudaStream { device, inner }
        }
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub unsafe fn inner(&self) -> cudaStream_t {
        self.inner
    }
}

#[derive(Debug)]
pub struct CudnnHandle {
    inner: cudnnHandle_t,
    stream: CudaStream,
}

impl Drop for CudnnHandle {
    fn drop(&mut self) {
        unsafe {
            cuda_set_device(self.device());
            cudnnDestroy(self.inner).unwrap()
        }
    }
}

impl CudnnHandle {
    pub fn new(stream: CudaStream) -> Self {
        unsafe {
            let mut inner = null_mut();
            cuda_set_device(stream.device());
            cudnnCreate(&mut inner as *mut _).unwrap();
            cudnnSetStream(inner, stream.inner()).unwrap();
            CudnnHandle { inner, stream }
        }
    }

    pub fn device(&self) -> i32 {
        self.stream.device()
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    pub unsafe fn inner(&mut self) -> cudnnHandle_t {
        self.inner
    }
}
