use std::ffi::c_void;
use std::ptr::null_mut;

use crate::bindings::{cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyKind, cudaMemset};
use crate::wrapper::status::Status;
use crate::wrapper::handle::Device;

#[derive(Debug)]
pub struct DeviceMem {
    dev_ptr: *mut c_void,
    size: usize,
    device: Device,
}

impl Drop for DeviceMem {
    fn drop(&mut self) {
        unsafe {
            self.device.switch_to();
            cudaFree(self.dev_ptr).unwrap()
        }
    }
}

impl DeviceMem {
    pub fn alloc(size: usize, device: Device) -> Self {
        unsafe {
            let mut dev_ptr = null_mut();
            device.switch_to();
            cudaMalloc(&mut dev_ptr as *mut _, size).unwrap();
            DeviceMem { dev_ptr, size, device }
        }
    }

    pub unsafe fn from_components(dev_ptr: *mut c_void, size: usize, device: Device) -> Self {
        DeviceMem { dev_ptr, size, device }
    }

    pub unsafe fn inner(&self) -> *mut c_void {
        self.dev_ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn fill_with_byte(&mut self, value: u8) {
        unsafe {
            self.device.switch_to();
            cudaMemset(self.dev_ptr, value as i32, self.size).unwrap()
        }
    }

    pub fn copy_from_host(&mut self, buffer: &[u8]) {
        assert_eq!(self.size, buffer.len(), "copy buffer size mismatch");
        unsafe {
            self.device.switch_to();
            cudaMemcpy(
                self.dev_ptr,
                buffer.as_ptr() as *const c_void,
                self.size,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            ).unwrap()
        }
    }

    pub fn copy_to_host(&self, buffer: &mut [u8]) {
        assert_eq!(self.size, buffer.len(), "copy buffer size mismatch");
        unsafe {
            self.device.switch_to();
            cudaMemcpy(
                buffer.as_mut_ptr() as *mut c_void,
                self.dev_ptr,
                self.size,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            ).unwrap()
        }
    }

    pub fn copy_from_device(&mut self, other: &DeviceMem) {
        assert_eq!(self.size, other.size, "copy buffer size mismatch");
        unsafe {
            self.device.switch_to();
            cudaMemcpy(
                self.dev_ptr,
                other.dev_ptr,
                self.size,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ).unwrap()
        }
    }
}