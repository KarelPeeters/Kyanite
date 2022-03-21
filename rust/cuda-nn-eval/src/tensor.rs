use bytemuck::{cast_slice, cast_slice_mut};

use cuda_sys::bindings::{cublasOperation_t, cudnnOpTensorOp_t};
use cuda_sys::wrapper::descriptor::{TensorDescriptor, TensorOpDescriptor};
use cuda_sys::wrapper::group::{MatMulArg, TensorOpArgs};
use cuda_sys::wrapper::handle::{CudnnHandle, Device};
use cuda_sys::wrapper::mem::device::DevicePtr;
use nn_graph::graph::SliceRange;

use crate::shape::StridedShape;

/// A tensor allocated on the device.
///
/// Cloning this type does not copy the underlying memory.
#[derive(Debug, Clone)]
pub struct DeviceTensor {
    pub ptr: DevicePtr,
    pub shape: StridedShape,
}

impl DeviceTensor {
    pub fn new(ptr: DevicePtr, shape: StridedShape) -> Self {
        DeviceTensor { ptr, shape }
    }

    pub fn alloc_simple(device: Device, shape: Vec<usize>) -> Self {
        let size = shape.iter().product::<usize>();
        let ptr = DevicePtr::alloc(device, size * 4);
        DeviceTensor::new(ptr, StridedShape::new_simple(shape))
    }

    pub fn device(&self) -> Device {
        self.ptr.device()
    }

    pub fn permute(&self, permutation: &[usize]) -> DeviceTensor {
        DeviceTensor::new(self.ptr.clone(), self.shape.permute(permutation))
    }

    pub fn view(&self, new_shape: Vec<usize>) -> DeviceTensor {
        let new_shape = self.shape.view(new_shape.clone()).unwrap();
        DeviceTensor::new(self.ptr.clone(), new_shape)
    }

    pub fn slice(&self, axis: usize, range: SliceRange) -> DeviceTensor {
        // use the new shape & strides (which only change along `axis`)
        let result_shape = self.shape.slice(axis, range);

        let offset = if result_shape.size() != 0 {
            // offset initial pointer to account for `start`
            result_shape.strides()[axis] * range.start as isize
        } else {
            0
        };

        let ptr = self.ptr.offset(4 * offset as isize);
        DeviceTensor::new(ptr, result_shape)
    }

    pub fn index(&self, axis: usize, index: usize) -> DeviceTensor {
        let mut new_shape = self.shape.shape().to_vec();
        new_shape.remove(axis);

        self.slice(axis, SliceRange::simple(index, index + 1)).view(new_shape)
    }

    pub fn flip(&self, axis: usize) -> DeviceTensor {
        // invert the axis stride
        let result_shape = self.shape.flip(axis);

        let axis_len = self.shape.shape()[axis];
        let offset = if self.shape.size() != 0 && axis_len != 0 {
            // offset so index 0 gets the last element along the axis
            (axis_len - 1) as isize * self.shape.strides()[axis]
        } else {
            0
        };

        let ptr = self.ptr.offset(4 * offset as isize);
        DeviceTensor::new(ptr, result_shape)
    }

    pub fn to_mat_mul_arg(&self) -> MatMulArg {
        assert_eq!(self.shape.rank(), 3);

        let inner_shape = StridedShape::new(self.shape.shape()[1..].to_vec(), self.shape.strides()[1..].to_vec());

        // whether the strides are col-major (true) or row-major (false)
        let col_major = if inner_shape.has_simple_strides() {
            false
        } else if inner_shape.permute(&[1, 0]).has_simple_strides() {
            true
        } else {
            panic!(
                "For now GPU matmul operand must be either col- or row-major, got {:?}",
                self
            )
        };

        let lead_axis = if col_major { 1 } else { 2 };

        MatMulArg {
            ptr: self.ptr.clone(),
            trans: if col_major {
                cublasOperation_t::CUBLAS_OP_N
            } else {
                cublasOperation_t::CUBLAS_OP_T
            },
            ld: self.shape.shape()[lead_axis] as i32,
            stride: self.shape.strides()[0] as i64,
        }
    }

    pub fn deep_clone(&self) -> DeviceTensor {
        let new = DeviceTensor::alloc_simple(self.device(), self.shape.shape().to_vec());
        unsafe {
            new.copy_from(self);
        }
        new
    }

    pub unsafe fn copy_simple_from_host(&self, buffer: &[f32]) {
        assert_eq!(
            self.shape.size(),
            buffer.len(),
            "Wrong buffer size for {:?}",
            self.shape
        );
        assert!(
            self.shape.has_simple_strides(),
            "Tensor must have simple strides for now, got {:?}",
            self.shape
        );
        assert_eq!(
            self.shape.size(),
            buffer.len(),
            "Wrong buffer size for {:?}",
            self.shape
        );
        self.ptr.copy_linear_from_host(cast_slice(buffer));
    }

    pub unsafe fn copy_simple_to_host(&self, buffer: &mut [f32]) {
        assert_eq!(
            self.shape.size(),
            buffer.len(),
            "Wrong buffer size for {:?}",
            self.shape
        );
        assert!(
            self.shape.has_simple_strides(),
            "Tensor must have simple strides, got {:?}",
            self.shape
        );
        self.ptr.copy_linear_to_host(cast_slice_mut(buffer));
    }

    pub fn copy_from_as_tensor_op(&self, other: &DeviceTensor) -> TensorOpArgs {
        assert_eq!(
            self.shape.shape(),
            other.shape.shape(),
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
            input_1_desc: other.shape.descriptor(),
            input_1_ptr: other.ptr.clone(),
            alpha_2: 0.0,
            input_2_desc: rhs_desc,
            input_2_ptr: other.ptr.clone(),
            beta: 0.0,
            output_desc: self.shape.descriptor(),
            output_ptr: self.ptr.clone(),
        }
    }

    pub unsafe fn copy_from(&self, other: &DeviceTensor) {
        assert_eq!(
            self.shape.shape(),
            other.shape.shape(),
            "Tensors must have the same shape: {:?} vs {:?}",
            self,
            other
        );

        if self.shape == other.shape && self.shape.has_dense_strides() {
            // if strides are dense and match we can just do a simple memcpy
            self.ptr.copy_linear_from_device(&other.ptr, self.shape.size() * 4)
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
            self.shape.size(),
            buffer.len(),
            "Wrong buffer size for {:?}",
            self.shape
        );

        if self.shape.has_simple_strides() {
            self.copy_simple_from_host(buffer);
        } else {
            let stage = DeviceTensor::alloc_simple(self.device(), self.shape.shape().to_vec());
            stage.copy_simple_from_host(buffer);
            self.copy_from(&stage);
        }
    }

    /// A (potentially) slower version of [Self::copy_to_host] that works for any strides,
    /// by potentially copying to an intermediate stage on the device.
    pub unsafe fn copy_to_host_staged(&self, buffer: &mut [f32]) {
        assert_eq!(
            self.shape.size(),
            buffer.len(),
            "Wrong buffer size for {:?}",
            self.shape
        );

        if self.shape.has_simple_strides() {
            self.copy_simple_to_host(buffer);
        } else {
            let stage = DeviceTensor::alloc_simple(self.device(), self.shape.shape().to_vec());
            stage.copy_from(self);
            stage.copy_simple_to_host(buffer);
        }
    }
}
