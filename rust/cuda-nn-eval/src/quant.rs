use bytemuck::cast_slice;
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

    pub unsafe fn copy_from_simple_tensor(&self, other: &DeviceTensor) {
        assert!(other.shape.has_simple_strides());
        assert_eq!(self.device(), other.device());

        let device = self.device();
        let stream = CudaStream::new(device);

        let outputs = [self.ptr.ptr() as usize];
        let outputs_device = device.alloc(8);
        outputs_device.copy_linear_from_host(cast_slice(&outputs));

        kernels::quantize(
            stream.inner(),
            1,
            self.size as i32,
            other.ptr.ptr() as *const f32,
            outputs_device.ptr() as *mut *mut u8,
        )
        .unwrap();

        stream.synchronize();
        // temporary device memory is deallocated here
    }
    pub unsafe fn copy_to_simple_tensor(&self, other: &DeviceTensor) {
        assert!(other.shape.has_simple_strides());
        assert_eq!(self.device(), other.device());

        let device = self.device();
        let stream = CudaStream::new(device);

        let inputs = [self.ptr.ptr() as usize];
        let inputs_device = device.alloc(8);
        inputs_device.copy_linear_from_host(cast_slice(&inputs));

        kernels::unquantize(
            stream.inner(),
            1,
            self.size as i32,
            inputs_device.ptr() as *const *const u8,
            other.ptr.ptr() as *mut f32,
        )
        .unwrap();

        stream.synchronize();
        // temporary device memory is deallocated here
    }
}
