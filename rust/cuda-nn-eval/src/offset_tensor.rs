use std::fmt::Debug;

use cuda_sys::bindings::cublasOperation_t;
use cuda_sys::wrapper::group::MatMulOperand;
use nn_graph::graph::SliceRange;

use crate::shape::{StridedShape, ViewError};

pub trait OffsetPtr: Debug + Clone {
    fn offset_bytes(self, offset: isize) -> Self;
}

/// A generic Tensor representation.
#[derive(Debug, Clone)]
pub struct PtrTensor<P> {
    shape: StridedShape,
    ptr: P,
}

impl<P> PtrTensor<P> {
    pub fn from_parts(ptr: P, shape: StridedShape) -> Self {
        PtrTensor { ptr, shape }
    }

    pub fn into_ptr(self) -> P {
        self.ptr
    }

    pub fn ptr(&self) -> &P {
        &self.ptr
    }

    pub fn shape(&self) -> &StridedShape {
        &self.shape
    }

    pub fn map_ptr<K>(self, f: impl FnOnce(P) -> K) -> PtrTensor<K> {
        PtrTensor::from_parts(f(self.ptr), self.shape)
    }
}

impl<P: OffsetPtr> PtrTensor<P> {
    fn offset(&self, offset: isize, shape: StridedShape) -> Self {
        Self::from_parts(self.ptr.clone().offset_bytes(4 * offset), shape)
    }

    pub fn permute(&self, permutation: &[usize]) -> Self {
        self.offset(0, self.shape.permute(permutation))
    }

    pub fn view(&self, new_shape: Vec<usize>) -> Result<Self, ViewError> {
        self.shape.view(new_shape).map(|shape| self.offset(0, shape))
    }

    pub fn broadcast(&self, new_shape: Vec<usize>) -> Self {
        self.offset(0, self.shape.broadcast(new_shape))
    }

    pub fn slice(&self, axis: usize, range: impl Into<SliceRange>) -> Self {
        let range = range.into();

        // use the new shape & strides (which only change along `axis`)
        let result_shape = self.shape.slice(axis, range);

        let offset = if result_shape.size() != 0 {
            // offset initial pointer to account for `start`
            result_shape.strides()[axis] * range.start as isize
        } else {
            0
        };

        self.offset(offset, result_shape)
    }

    pub fn index(&self, axis: usize, index: usize) -> Self {
        let mut new_shape = self.shape.shape().to_vec();
        new_shape.remove(axis);

        self.slice(axis, SliceRange::simple(index, index + 1))
            .view(new_shape)
            .unwrap()
    }

    pub fn flip(&self, axis: usize) -> Self {
        // invert the axis stride
        let result_shape = self.shape.flip(axis);

        let axis_len = self.shape.shape()[axis];
        let offset = if self.shape.size() != 0 && axis_len != 0 {
            // offset so index 0 gets the last element along the axis
            (axis_len - 1) as isize * self.shape.strides()[axis]
        } else {
            0
        };

        self.offset(offset, result_shape)
    }

    pub fn repeat_unary(&self, axis: usize, count: usize) -> Self {
        let result_shape = self.shape.repeat_unary(axis, count);
        self.offset(0, result_shape)
    }
}

impl<P: Clone> PtrTensor<P> {
    //TODO move this somewhere else, this is pretty random
    pub fn to_mat_mul_arg(&self) -> MatMulOperand<P> {
        assert_eq!(self.shape().rank(), 3);

        // prefer col-major in case of a tie, since cublas likes that more
        let (trans, lead_axis) = if self.shape.strides()[1] == 1 {
            (cublasOperation_t::CUBLAS_OP_N, 2)
        } else if self.shape.strides()[2] == 1 {
            (cublasOperation_t::CUBLAS_OP_T, 1)
        } else {
            panic!(
                "GPU matmul operand must be either col- or row-dense, got {:?}",
                self.shape
            )
        };

        MatMulOperand {
            ptr: self.ptr().clone(),
            trans,
            ld: self.shape.strides()[lead_axis] as i32,
            stride: self.shape().strides()[0] as i64,
        }
    }
}
