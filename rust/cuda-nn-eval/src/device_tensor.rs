use bytemuck::{cast_slice, cast_slice_mut};

use crate::offset_tensor::{OffsetPtr, PtrTensor};
use cuda_sys::bindings::cudnnOpTensorOp_t;
use cuda_sys::wrapper::descriptor::{TensorDescriptor, TensorOpDescriptor};
use cuda_sys::wrapper::group::TensorOpArgs;
use cuda_sys::wrapper::handle::{CudnnHandle, Device};
use cuda_sys::wrapper::mem::device::DevicePtr;

use crate::shape::StridedShape;

pub type DeviceTensor = PtrTensor<DevicePtr>;

impl OffsetPtr for DevicePtr {
    fn offset_bytes(self, offset: isize) -> Self {
        DevicePtr::offset_bytes(self, offset)
    }
}

impl DeviceTensor {
    pub fn alloc_simple(device: Device, shape: Vec<usize>) -> Self {
        let shape = StridedShape::new_simple(shape);
        let ptr = DevicePtr::alloc(device, shape.size() * 4);
        DeviceTensor::from_parts(ptr, shape)
    }

    pub fn device(&self) -> Device {
        self.ptr().device()
    }

    pub fn deep_clone(&self) -> DeviceTensor {
        let new = DeviceTensor::alloc_simple(self.device(), self.strided_shape().shape().to_vec());
        unsafe {
            new.copy_from(self);
        }
        new
    }

    pub unsafe fn copy_simple_from_host(&self, buffer: &[f32]) {
        assert_eq!(
            buffer.len(),
            self.strided_shape().size(),
            "Wrong buffer size {} for {:?}",
            buffer.len(),
            self.strided_shape()
        );
        assert!(
            self.strided_shape().has_simple_strides(),
            "Tensor must have simple strides for now, got {:?}",
            self.strided_shape()
        );
        assert_eq!(
            buffer.len(),
            self.strided_shape().size(),
            "Wrong buffer size for {:?}",
            self.strided_shape()
        );
        self.ptr().copy_linear_from_host(cast_slice(buffer));
    }

    pub unsafe fn copy_simple_to_host(&self, buffer: &mut [f32]) {
        assert_eq!(
            self.strided_shape().size(),
            buffer.len(),
            "Wrong buffer size {} for {:?}",
            buffer.len(),
            self.strided_shape()
        );
        assert!(
            self.strided_shape().has_simple_strides(),
            "Tensor must have simple strides, got {:?}",
            self.strided_shape()
        );
        self.ptr().copy_linear_to_host(cast_slice_mut(buffer));
    }

    pub fn copy_from_as_tensor_op(&self, other: &DeviceTensor) -> TensorOpArgs {
        assert_eq!(
            self.strided_shape().shape(),
            other.strided_shape().shape(),
            "Tensors must have the same shape: {:?} vs {:?}",
            self,
            other
        );

        let op_desc = TensorOpDescriptor::new(cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD);

        // the value of the RHS tensor does not matter (alpha_2 = 0), so we reuse the input to avoid an allocation
        // (this should also work if both input and output are empty)
        let rhs_desc = TensorDescriptor::new(vec![1, 1, 1, 1], vec![1, 1, 1, 1]);

        TensorOpArgs {
            op_desc,
            alpha_1: 1.0,
            input_1_desc: other.strided_shape().descriptor(),
            input_1_ptr: other.ptr().clone(),
            alpha_2: 0.0,
            input_2_desc: rhs_desc,
            input_2_ptr: other.ptr().clone(),
            beta: 0.0,
            output_desc: self.strided_shape().descriptor(),
            output_ptr: self.ptr().clone(),
        }
    }

    pub unsafe fn copy_from(&self, other: &DeviceTensor) {
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
                .copy_linear_from_device(&other.ptr(), self.strided_shape().size() * 4)
        } else {
            // otherwise use the TensorOp restride trick
            let handle = CudnnHandle::new(self.device());
            self.copy_from_as_tensor_op(&other).run(&handle);
            handle.stream().synchronize();
        }
    }

    /// A (potentially) slower version of [Self::copy_from_host] that works for any strides,
    /// by potentially copying to an intermediate stage on the device.
    pub unsafe fn copy_from_host_staged(&self, buffer: &[f32]) {
        assert_eq!(
            self.strided_shape().size(),
            buffer.len(),
            "Wrong buffer size for {:?}",
            self.strided_shape()
        );

        if self.strided_shape().has_simple_strides() {
            self.copy_simple_from_host(buffer);
        } else {
            let stage = DeviceTensor::alloc_simple(self.device(), self.strided_shape().shape().to_vec());
            stage.copy_simple_from_host(buffer);
            self.copy_from(&stage);
        }
    }

    /// A (potentially) slower version of [Self::copy_to_host] that works for any strides,
    /// by potentially copying to an intermediate stage on the device.
    pub unsafe fn copy_to_host_staged(&self, buffer: &mut [f32]) {
        assert_eq!(
            self.strided_shape().size(),
            buffer.len(),
            "Wrong buffer size for {:?}",
            self.strided_shape()
        );

        if self.strided_shape().has_simple_strides() {
            self.copy_simple_to_host(buffer);
        } else {
            let stage = DeviceTensor::alloc_simple(self.device(), self.strided_shape().shape().to_vec());
            stage.copy_from(self);
            stage.copy_simple_to_host(buffer);
        }
    }
}
