use std::ptr::null_mut;

use crate::bindings::{cudaEventCreate, cudaEventDestroy, cudaEventElapsedTime, cudaEventSynchronize, cudaEvent_t};
use crate::wrapper::status::Status;

#[derive(Debug)]
pub struct CudaEvent(cudaEvent_t);

impl CudaEvent {
    pub fn new() -> Self {
        unsafe {
            let mut inner = null_mut();
            cudaEventCreate(&mut inner as *mut _).unwrap();
            CudaEvent(inner)
        }
    }

    /// Return the elapsed time since `start` _in seconds_.
    pub fn time_elapsed_since(&self, start: &CudaEvent) -> f32 {
        unsafe {
            let mut result: f32 = 0.0;
            cudaEventElapsedTime(&mut result as *mut _, start.inner(), self.inner()).unwrap();
            result / 1000.0
        }
    }

    pub fn synchronize(&self) {
        unsafe { cudaEventSynchronize(self.inner()).unwrap() }
    }

    pub unsafe fn inner(&self) -> cudaEvent_t {
        self.0
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe { cudaEventDestroy(self.0).unwrap_in_drop() }
    }
}
