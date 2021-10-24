use std::ffi::c_void;
use std::ptr::null_mut;
use std::rc::Rc;

use crate::bindings::{cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyAsync, cudaMemcpyKind, cudaMemset};
use crate::wrapper::handle::{CudaStream, Device};
use crate::wrapper::status::Status;

/// A view on (part of) a device buffer.
/// Multiple views can share the same underlying buffer,
/// consequently all functions that read or write from this view are marked `unsafe`.
/// The underlying buffer is freed when all views on it have been dropped.
#[derive(Debug)]
pub struct DeviceMem {
    inner: Rc<DeviceMemInner>,
    offset: isize,
    len: usize,
}

/// The underlying device buffer.
#[derive(Debug)]
struct DeviceMemInner {
    device: Device,
    dev_ptr: *mut c_void,
}

impl Drop for DeviceMemInner {
    fn drop(&mut self) {
        unsafe {
            self.device.switch_to();
            cudaFree(self.dev_ptr).unwrap()
        }
    }
}

impl DeviceMem {
    pub fn alloc(size_in_bytes: usize, device: Device) -> Self {
        let inner = unsafe {
            let mut dev_ptr = null_mut();
            device.switch_to();
            cudaMalloc(&mut dev_ptr as *mut _, size_in_bytes).unwrap();

            DeviceMemInner { device, dev_ptr }
        };

        DeviceMem {
            inner: Rc::new(inner),
            offset: 0,
            len: size_in_bytes,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn device(&self) -> Device {
        self.inner.device
    }

    /// Return a shallow clone of this value, the inner memory will be shared.
    pub fn view(&self) -> Self {
        DeviceMem {
            inner: Rc::clone(&self.inner),
            offset: self.offset,
            len: self.len,
        }
    }

    pub fn slice(&self, start: usize, len: usize) -> DeviceMem {
        assert!(
            start < self.len && start + len <= self.len,
            "Slice indices must be in bounds, got start {} len {} for mem of len {}",
            start, len, self.len,
        );

        DeviceMem {
            inner: Rc::clone(&self.inner),
            offset: self.offset + start as isize,
            len,
        }
    }

    pub unsafe fn ptr(&self) -> *mut c_void {
        self.inner.dev_ptr.offset(self.offset)
    }

    pub unsafe fn fill_with_byte(&self, value: u8) {
        self.inner.device.switch_to();
        cudaMemset(self.ptr(), value as i32, self.len).unwrap()
    }

    pub unsafe fn copy_from_host_async(&self, buffer: &[u8], stream: &mut CudaStream) {
        assert_eq!(self.len, buffer.len(), "copy buffer size mismatch");
        self.inner.device.switch_to();
        cudaMemcpyAsync(
            self.ptr(),
            buffer.as_ptr() as *const c_void,
            self.len,
            cudaMemcpyKind::cudaMemcpyHostToDevice,
            stream.inner(),
        ).unwrap()
    }

    pub unsafe fn copy_to_host_async(&self, buffer: &mut [u8], stream: &mut CudaStream) {
        assert_eq!(self.len, buffer.len(), "copy buffer size mismatch");

        self.inner.device.switch_to();
        cudaMemcpyAsync(
            buffer.as_mut_ptr() as *mut c_void,
            self.ptr(),
            self.len,
            cudaMemcpyKind::cudaMemcpyDeviceToHost,
            stream.inner(),
        ).unwrap()
    }

    pub unsafe fn copy_from_host(&self, buffer: &[u8]) {
        assert_eq!(self.len, buffer.len(), "copy buffer size mismatch");

        self.inner.device.switch_to();
        cudaMemcpy(
            self.ptr(),
            buffer.as_ptr() as *const c_void,
            self.len,
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        ).unwrap()
    }

    pub unsafe fn copy_to_host(&self, buffer: &mut [u8]) {
        assert_eq!(self.len, buffer.len(), "copy buffer size mismatch");

        self.inner.device.switch_to();
        cudaMemcpy(
            buffer.as_mut_ptr() as *mut c_void,
            self.ptr(),
            self.len,
            cudaMemcpyKind::cudaMemcpyDeviceToHost,
        ).unwrap()
    }

    pub unsafe fn copy_from_device(&self, other: &DeviceMem) {
        assert_eq!(self.len, other.len, "Buffer size mismatch");
        assert_eq!(self.inner.device, other.inner.device, "Buffers are on different devices");

        self.inner.device.switch_to();
        cudaMemcpy(
            self.ptr(),
            other.ptr(),
            self.len,
            cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        ).unwrap()
    }
}