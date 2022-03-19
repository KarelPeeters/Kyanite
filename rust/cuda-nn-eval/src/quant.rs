use cuda_sys::wrapper::handle::{CudaStream, Device};
use cuda_sys::wrapper::mem::device::DevicePtr;
use cuda_sys::wrapper::status::Status;

use crate::kernels;
use crate::tensor::DeviceTensor;

#[derive(Debug, Clone)]
pub struct QuantizedStorage {
    ptr: DevicePtr,
    size: usize,
}

impl QuantizedStorage {
    pub fn alloc(device: Device, size: usize) -> QuantizedStorage {
        let ptr = device.alloc(size);
        QuantizedStorage { ptr, size }
    }

    pub fn device(&self) -> Device {
        self.ptr.device()
    }

    pub fn ptr(&self) -> &DevicePtr {
        &self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub unsafe fn launch_copy_from_simple_tensor(&self, other: &DeviceTensor, stream: &CudaStream) {
        assert!(other.shape.has_simple_strides());
        assert_eq!(self.device(), other.device());

        kernels::quantize(
            stream.inner(),
            self.size as i32,
            other.ptr.ptr() as *const f32,
            self.ptr.ptr() as *mut u8,
        )
        .unwrap()
    }
    pub unsafe fn launch_copy_to_simple_tensor(&self, other: &DeviceTensor, stream: &CudaStream) {
        assert!(other.shape.has_simple_strides());
        assert_eq!(self.device(), other.device());

        kernels::unquantize(
            stream.inner(),
            self.size as i32,
            self.ptr.ptr() as *const u8,
            other.ptr.ptr() as *mut f32,
        )
        .unwrap()
    }
}
