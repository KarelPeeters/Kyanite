use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::Arc;

use crate::bindings::{cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyKind};
use crate::wrapper::handle::Device;
use crate::wrapper::status::Status;

/// A pointer pointing somewhere inside of a [DeviceBuffer].
/// The inner [DeviceBuffer] is automatically freed when there are no [DevicePtr] any more that refer to it.
/// Since the memory may be shared all accessor methods are marked unsafe.
///
/// Cloning this type does not copy the underlying memory.
#[derive(Debug, Clone)]
pub struct DevicePtr {
    buffer: Arc<DeviceBuffer>,
    offset: isize,
}

/// A device allocation as returned by cudaMalloc.
#[derive(Debug)]
struct DeviceBuffer {
    device: Device,
    base_ptr: *mut c_void,
    len_bytes: isize,
}

// TODO is this correct? We've don't even attempt device-side memory safety, but can this cause cpu-side issues?
// TODO should we implement Sync? It's probably never a good idea to actually share pointers between threads...
unsafe impl Send for DeviceBuffer {}

unsafe impl Sync for DeviceBuffer {}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.switch_to();
            cudaFree(self.base_ptr).unwrap_in_drop()
        }
    }
}

impl DevicePtr {
    pub fn alloc(device: Device, len_bytes: usize) -> Self {
        unsafe {
            let mut device_ptr = null_mut();

            device.switch_to();
            cudaMalloc(&mut device_ptr as *mut _, len_bytes).unwrap();

            let inner = DeviceBuffer {
                device,
                base_ptr: device_ptr,
                len_bytes: len_bytes as isize,
            };
            DevicePtr {
                buffer: Arc::new(inner),
                offset: 0,
            }
        }
    }

    pub fn device(&self) -> Device {
        self.buffer.device
    }

    pub fn offset(&self, offset: isize) -> DevicePtr {
        let new_offset = self.offset + offset;

        assert!(
            (0..self.buffer.len_bytes as isize).contains(&new_offset),
            "Offset {} is out of range on {:?}",
            offset,
            self
        );

        DevicePtr {
            buffer: Arc::clone(&self.buffer),
            offset: new_offset,
        }
    }

    pub unsafe fn ptr(&self) -> *mut c_void {
        self.buffer.base_ptr.offset(self.offset)
    }

    pub unsafe fn copy_linear_from_host(&self, buffer: &[u8]) {
        self.assert_linear_in_bounds(buffer.len());

        self.device().switch_to();
        cudaMemcpy(
            self.ptr(),
            buffer as *const _ as *const _,
            buffer.len(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        )
        .unwrap();
    }

    pub unsafe fn copy_linear_to_host(&self, buffer: &mut [u8]) {
        self.assert_linear_in_bounds(buffer.len());

        self.device().switch_to();
        cudaMemcpy(
            buffer as *mut _ as *mut _,
            self.ptr(),
            buffer.len(),
            cudaMemcpyKind::cudaMemcpyDeviceToHost,
        )
        .unwrap();
    }

    pub unsafe fn copy_linear_from_device(&self, other: &DevicePtr, len_bytes: usize) {
        assert_eq!(
            self.device(),
            other.device(),
            "Can only copy between tensors on the same device"
        );

        self.assert_linear_in_bounds(len_bytes);
        other.assert_linear_in_bounds(len_bytes);

        self.device().switch_to();
        cudaMemcpy(
            self.ptr(),
            other.ptr(),
            len_bytes,
            cudaMemcpyKind::cudaMemcpyDeviceToDevice,
        )
        .unwrap();
    }

    fn assert_linear_in_bounds(&self, len: usize) {
        assert!(
            (self.offset + len as isize) <= self.buffer.len_bytes,
            "Linear slice with length {} out of bounds for {:?}",
            len,
            self
        );
    }
}
