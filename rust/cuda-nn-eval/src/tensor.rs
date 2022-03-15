use bytemuck::{cast_slice, cast_slice_mut};

use cuda_sys::bindings::cublasOperation_t;
use cuda_sys::wrapper::group::MatMulArg;
use cuda_sys::wrapper::handle::Device;
use cuda_sys::wrapper::mem::device::DevicePtr;

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

    pub fn alloc(device: Device, shape: Vec<usize>) -> Self {
        let size = shape.iter().product::<usize>();
        let ptr = DevicePtr::alloc(device, size * 4);
        DeviceTensor::new(ptr, StridedShape::new_simple(shape))
    }

    pub fn permute(&self, permutation: &[usize]) -> DeviceTensor {
        DeviceTensor::new(self.ptr.clone(), self.shape.permute(permutation))
    }

    pub fn slice(&self, axis: usize, start: usize, end: usize) -> DeviceTensor {
        // Steps to slice a tensor:
        //  * use the new shape
        //  * keep the old strides
        //  * offset initial pointer to account for `start`
        //  * limit the buffer length based on the new size
        let result_shape = self.shape.slice(axis, start, end);

        let start_bytes = result_shape.strides()[axis] * start * 4;
        let mem = self.ptr.offset(start_bytes as isize);

        DeviceTensor::new(mem, result_shape)
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

    pub unsafe fn copy_from(&self, other: &DeviceTensor) {
        assert_eq!(
            self.shape, other.shape,
            "Both tensors must have the same shape and stride for now"
        );
        // necessary to ensure "in-between" data does not get overriden
        assert!(
            self.shape.has_simple_strides(),
            "Tensors must have simple stride for now, got {:?}",
            self.shape
        );
        self.ptr.copy_linear_from_device(&other.ptr, self.shape.strided_size());
    }

    pub unsafe fn copy_from_host(&self, buffer: &[f32]) {
        assert!(
            self.shape.has_simple_strides(),
            "Tensor must have simple stride for now, got {:?}",
            self.shape
        );
        self.ptr.copy_linear_from_host(cast_slice(buffer));
    }

    pub unsafe fn copy_to_host(&self, buffer: &mut [f32]) {
        assert!(
            self.shape.has_simple_strides(),
            "Tensor must have simple stride for now, got {:?}",
            self.shape
        );
        self.ptr.copy_linear_to_host(cast_slice_mut(buffer));
    }
}
