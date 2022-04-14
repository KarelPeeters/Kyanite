use bytemuck::cast_slice;

use cuda_sys::wrapper::event::CudaEvent;
use cuda_sys::wrapper::handle::{CudaStream, Device};
use cuda_sys::wrapper::mem::device::DevicePtr;
use cuda_sys::wrapper::status::Status;

use crate::device_tensor::DeviceTensor;
use crate::kernels;

#[derive(Debug)]
pub struct BatchQuantizer {
    max_batch_size: usize,
    pointers: Vec<usize>,
    pointers_device: DevicePtr,
    last_event: Option<CudaEvent>,
}

impl BatchQuantizer {
    pub fn new(device: Device, max_batch_size: usize) -> BatchQuantizer {
        BatchQuantizer {
            max_batch_size,
            pointers: vec![0; max_batch_size],
            pointers_device: device.alloc(8 * max_batch_size),
            last_event: None,
        }
    }

    pub fn device(&self) -> Device {
        self.pointers_device.device()
    }

    //TODO maybe it's enough to synchronize only another stream? careful about the CPU if they become async though!
    /// Wait for previous quantization to complete if any.
    /// This is necessary to ensure the device memory is no longer being used by the previous kernel launch.
    pub fn synchronize(&mut self) {
        if let Some(event) = self.last_event.take() {
            unsafe {
                event.synchronize();
            }
        }
    }

    unsafe fn prepare_pointers<'a>(
        &mut self,
        tensor: &DeviceTensor,
        quantized: impl Iterator<Item = &'a QuantizedStorage>,
    ) -> (usize, usize) {
        assert!(tensor.shape().has_simple_strides());
        assert!(tensor.shape().rank() >= 2, "Tensor must have at least rank 2");
        assert_eq!(tensor.device(), self.device());

        let batch_size = tensor.shape().shape()[0];
        let size = tensor.shape().shape()[1..].iter().product::<usize>();

        // collect pointers
        let mut q_batch_size = 0;
        self.pointers.clear();

        for q in quantized {
            assert_eq!(
                q.size,
                size,
                "Size mismatch: got size {}, but expected {} for tensor shape {:?}",
                q.size,
                size,
                tensor.shape()
            );
            assert_eq!(q.device(), self.device());

            self.pointers.push(q.ptr.ptr() as usize);
            q_batch_size += 1;
        }

        // it it not our responsibility to support a smaller batch size here,
        // the user should just slice the tensor instead
        assert_eq!(
            batch_size,
            q_batch_size,
            "Batch size mismatch, tensor {:?} but got {} quantized storages",
            tensor.shape().shape(),
            q_batch_size
        );

        assert!(
            batch_size <= self.max_batch_size,
            "Batch size too large, got {} but max {}",
            batch_size,
            self.max_batch_size
        );

        (batch_size, size)
    }

    pub unsafe fn launch_quantize<'a>(
        &mut self,
        stream: &CudaStream,
        source: &DeviceTensor,
        dest: impl Iterator<Item = &'a QuantizedStorage>,
    ) {
        self.synchronize();

        let (batch_size, size) = self.prepare_pointers(source, dest);

        // TODO use async copy (form pinned mem)?
        self.pointers_device.copy_linear_from_host(cast_slice(&self.pointers));

        kernels::quantize(
            stream.inner(),
            batch_size as i32,
            size as i32,
            source.ptr().ptr() as *const f32,
            self.pointers_device.ptr() as *mut *mut u8,
        )
        .unwrap();

        self.last_event = Some(stream.record_new_event());
    }

    pub unsafe fn launch_unquantize<'a>(
        &mut self,
        stream: &CudaStream,
        source: impl Iterator<Item = &'a QuantizedStorage>,
        dest: &DeviceTensor,
    ) {
        self.synchronize();

        let (batch_size, size) = self.prepare_pointers(dest, source);

        // TODO use async copy (form pinned mem)?
        self.pointers_device.copy_linear_from_host(cast_slice(&self.pointers));

        kernels::unquantize(
            stream.inner(),
            batch_size as i32,
            size as i32,
            self.pointers_device.ptr() as *const *const u8,
            dest.ptr().ptr() as *mut f32,
        )
        .unwrap();

        self.last_event = Some(stream.record_new_event());
    }
}

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

    pub unsafe fn quantize_from(&self, other: &DeviceTensor) {
        let device = self.device();
        let stream = CudaStream::new(device);

        let mut quantizer = BatchQuantizer::new(device, 1);
        quantizer.launch_quantize(&stream, other, std::iter::once(self));

        stream.synchronize();
        // quantizer is deallocated here, after operations have completed
    }
    pub unsafe fn unquantize_to(&self, other: &DeviceTensor) {
        let device = self.device();
        let stream = CudaStream::new(device);

        let mut quantizer = BatchQuantizer::new(device, 1);
        quantizer.launch_unquantize(&stream, std::iter::once(self), other);

        stream.synchronize();
        // quantizer is deallocated here, after operations have completed
    }
}
