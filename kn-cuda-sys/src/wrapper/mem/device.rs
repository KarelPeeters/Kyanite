use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::Arc;

use crate::bindings::{cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyAsync, cudaMemcpyKind};
use crate::wrapper::handle::{CudaStream, Device};
use crate::wrapper::mem::pinned::PinnedMem;
use crate::wrapper::status::Status;

/// A reference-counted pointer into a [DeviceBuffer]. The buffer cannot be constructed directly,
/// instead it can only be created by allocating a new [DevicePtr] with [DevicePtr::alloc].
///
/// The inner [DeviceBuffer] is automatically freed when there are no [DevicePtr] any more that refer to it.
/// Since the memory may be shared all accessor methods are marked unsafe.
///
/// Cloning this type does not copy the underlying memory, but only increases the reference count.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct DevicePtr {
    buffer: Arc<DeviceBuffer>,
    offset: isize,
}

// TODO it's a bit weird that this is public without being able to construct it,
// but it's useful to let the user know it exists in docs.
/// A device allocation as returned by cudaMalloc.
#[derive(Debug, Eq, PartialEq, Hash)]
pub struct DeviceBuffer {
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

    pub fn offset_bytes(self, offset: isize) -> DevicePtr {
        let new_offset = self.offset + offset;

        if self.buffer.len_bytes == 0 {
            assert_eq!(offset, 0, "Non-zero offset not allowed on empty buffer");
        } else {
            assert!(
                (0..self.buffer.len_bytes).contains(&new_offset),
                "Offset {} is out of range on {:?}",
                offset,
                self
            );
        }

        DevicePtr {
            buffer: self.buffer,
            offset: new_offset,
        }
    }

    pub unsafe fn ptr(&self) -> *mut c_void {
        self.buffer.base_ptr.offset(self.offset)
    }

    /// The number of `DevicePtr` sharing the underlying buffer that are still alive.
    pub fn shared_count(&self) -> usize {
        Arc::strong_count(&self.buffer)
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

    pub unsafe fn copy_linear_from_host_async(&self, buffer: &PinnedMem, stream: &CudaStream) {
        self.assert_linear_in_bounds(buffer.len_bytes());

        self.device().switch_to();
        cudaMemcpyAsync(
            self.ptr(),
            buffer.ptr(),
            buffer.len_bytes(),
            cudaMemcpyKind::cudaMemcpyDeviceToHost,
            stream.inner(),
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

    pub unsafe fn copy_linear_to_host_async(&self, buffer: &mut PinnedMem, stream: &CudaStream) {
        self.assert_linear_in_bounds(buffer.len_bytes());

        self.device().switch_to();
        cudaMemcpyAsync(
            buffer.ptr(),
            self.ptr(),
            buffer.len_bytes(),
            cudaMemcpyKind::cudaMemcpyDeviceToHost,
            stream.inner(),
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
