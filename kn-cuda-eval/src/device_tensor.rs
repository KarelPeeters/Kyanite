use bytemuck::{cast_slice, cast_slice_mut};

use kn_cuda_sys::wrapper::handle::{CudaStream, Device};
use kn_cuda_sys::wrapper::mem::device::DevicePtr;
use kn_graph::dispatch_dtensor;
use kn_graph::dtype::{DTensor, DType};

use crate::autokernel::scalar::ScalarKernel;
use crate::offset_tensor::{OffsetPtr, PtrTensor};
use crate::shape::StridedShape;
use crate::step::{OperandKind, ScalarOpArgs};

pub type DeviceTensor = PtrTensor<DevicePtr>;

impl OffsetPtr for DevicePtr {
    fn offset_bytes(self, offset: isize) -> Self {
        DevicePtr::offset_bytes(self, offset)
    }
}

impl DeviceTensor {
    pub fn alloc_simple(device: Device, shape: Vec<usize>, dtype: DType) -> Self {
        let shape = StridedShape::new_simple(shape);
        let ptr = DevicePtr::alloc(device, shape.size() * dtype.size().bytes());
        DeviceTensor::from_parts(ptr, shape, dtype)
    }

    pub fn device(&self) -> Device {
        self.ptr().device()
    }
    
    pub fn alloc_simple_init(device: Device, value: &DTensor) -> Self {
        let tensor = Self::alloc_simple(device, value.shape().to_vec(), value.dtype());

        // copy values
        dispatch_dtensor!(value, |T, _f, inner| {
            let inner = inner.as_standard_layout();
            let bytes = cast_slice::<T, u8>(inner.as_slice().unwrap());
            unsafe {
                tensor.copy_simple_from_host(bytes);
            }
        });
        
        tensor
    }

    pub fn deep_clone(&self) -> DeviceTensor {
        let new = DeviceTensor::alloc_simple(self.device(), self.strided_shape().shape().to_vec(), self.dtype());
        unsafe {
            new.copy_from(self);
        }
        new
    }

    pub unsafe fn copy_simple_from_host(&self, buffer: &[u8]) {
        assert!(
            self.strided_shape().has_simple_strides(),
            "Tensor must have simple strides, got {:?}",
            self,
        );
        assert_eq!(
            buffer.len(),
            self.dense_size_bytes(),
            "Wrong buffer size {} for {:?}",
            buffer.len(),
            self,
        );
        self.ptr().copy_linear_from_host(cast_slice(buffer));
    }

    pub unsafe fn copy_simple_to_host(&self, buffer: &mut [u8]) {
        assert!(
            self.strided_shape().has_simple_strides(),
            "Tensor must have simple strides, got {:?}",
            self,
        );
        assert_eq!(
            self.dense_size_bytes(),
            buffer.len(),
            "Wrong buffer size {} for {:?}",
            buffer.len(),
            self,
        );
        self.ptr().copy_linear_to_host(cast_slice_mut(buffer));
    }

    // TODO ideally we would decay to memcpy if possible
    //   but callers can already do that, this is this fallback!
    pub fn copy_from_as_scalar_op(&self, other: &DeviceTensor) -> ScalarOpArgs<DevicePtr> {
        assert_eq!(self.device(), other.device(), "Tensors must be on the same device");
        assert_eq!(self.dtype(), other.dtype(), "Tensors must have the same dtype");
        let device = self.device();
        let dtype = self.dtype();

        assert_eq!(
            self.strided_shape().shape(),
            other.strided_shape().shape(),
            "Tensors must have the same shape: {:?} vs {:?}",
            self,
            other
        );

        let dtype_str = dtype.as_c_str();
        let kernel = ScalarKernel::new_for_shapes(
            device,
            "*x0 = *x1",
            &[self.strided_shape().clone(), other.strided_shape().clone()],
            vec![dtype_str.to_owned(), dtype_str.to_owned()],
        );

        ScalarOpArgs {
            kernel,
            operands: vec![self.clone(), other.clone()],
            operand_kinds: vec![OperandKind::Out, OperandKind::In]
        }
    }

    pub unsafe fn copy_from(&self, other: &DeviceTensor) {
        assert_eq!(self.dtype(), other.dtype(), "Tensors must have the same dtype");
        let dtype = self.dtype();
        assert_eq!(self.device(), other.device(), "Tensors must be on the same device");
        let device = self.device();

        assert_eq!(
            self.strided_shape().shape(),
            other.strided_shape().shape(),
            "Tensors must have the same shape: {:?} vs {:?}",
            self,
            other
        );

        if self.strided_shape() == other.strided_shape() && self.strided_shape().has_dense_strides() {
            // if strides are dense and match we can just do a simple memcpy
            self.ptr()
                .copy_linear_from_device(&other.ptr(), self.strided_shape().size() * dtype.size().bytes())
        } else {
            // otherwise use the TensorOp restride trick
            let stream = CudaStream::new(device);
            self.copy_from_as_scalar_op(&other).run(&stream);
            stream.synchronize();
        }
    }

    /// A (potentially) slower version of [Self::copy_simple_from_host] that works for any strides,
    /// by potentially copying to an intermediate stage on the device.
    pub unsafe fn copy_from_host_staged(&self, buffer: &[u8]) {
        assert_eq!(self.dtype(), DType::F32, "Only f32 is supported for now");

        assert_eq!(
            self.strided_shape().size(),
            buffer.len(),
            "Wrong buffer size for {:?}",
            self.strided_shape()
        );

        if self.strided_shape().has_simple_strides() {
            self.copy_simple_from_host(buffer);
        } else {
            let stage = DeviceTensor::alloc_simple(self.device(), self.strided_shape().shape().to_vec(), self.dtype());
            stage.copy_simple_from_host(buffer);
            self.copy_from(&stage);
        }
    }

    /// A (potentially) slower version of [Self::copy_simple_to_host] that works for any strides,
    /// by potentially copying to an intermediate stage on the device.
    pub unsafe fn copy_to_host_staged(&self, buffer: &mut [u8]) {
        assert_eq!(self.dtype(), DType::F32, "Only f32 is supported for now");

        assert_eq!(
            self.strided_shape().size(),
            buffer.len(),
            "Wrong buffer size for {:?}",
            self.strided_shape()
        );

        if self.strided_shape().has_simple_strides() {
            self.copy_simple_to_host(buffer);
        } else {
            let stage = DeviceTensor::alloc_simple(self.device(), self.strided_shape().shape().to_vec(), self.dtype());
            stage.copy_from(self);
            stage.copy_simple_to_host(buffer);
        }
    }
}
