use std::ffi::c_void;
use std::ptr::null_mut;

use crate::bindings::{cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyKind, cudaMemset};
use crate::wrapper::status::Status;

/// TODO this does not currently support multiple devices!
///   because that's really ugly in cuda, it will require cudaSetDevice everywhere
///   (yay for mutable global state)
pub struct DeviceMem {
    dev_ptr: *mut c_void,
    size: usize,
}

impl Drop for DeviceMem {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.dev_ptr).unwrap()
        }
    }
}

impl DeviceMem {
    pub fn alloc(size: usize) -> Self {
        unsafe {
            let mut dev_ptr = null_mut();
            cudaMalloc(&mut dev_ptr as *mut _, size).unwrap();
            DeviceMem { dev_ptr, size }
        }
    }

    pub unsafe fn inner(&self) -> *mut c_void {
        self.dev_ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn fill_with_byte(&mut self, value: u8) {
        unsafe {
            //TODO is this async?
            cudaMemset(self.dev_ptr, value as i32, self.size).unwrap()
        }
    }

    pub fn copy_from_host(&mut self, buffer: &[u8]) {
        assert_eq!(self.size, buffer.len());
        unsafe {
            //TODO is this async?
            cudaMemcpy(
                self.dev_ptr,
                buffer.as_ptr() as *const c_void,
                self.size,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            ).unwrap()
        }
    }

    pub fn copy_to_host(&self, buffer: &mut [u8]) {
        assert_eq!(self.size, buffer.len());
        unsafe {
            //TODO is this async?
            cudaMemcpy(
                buffer.as_mut_ptr() as *mut c_void,
                self.dev_ptr,
                self.size,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            ).unwrap()
        }
    }

    pub fn copy_from_device(&mut self, other: &DeviceMem) {
        assert_eq!(self.size, other.size);
        unsafe {
            //TODO is this async?
            cudaMemcpy(
                self.dev_ptr,
                other.dev_ptr,
                self.size,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ).unwrap()
        }
    }
}