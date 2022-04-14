use crate::shape::StridedShape;
use nn_graph::graph::SliceRange;
use std::fmt::Debug;

pub trait OffsetPtr: Debug + Clone {
    fn offset_bytes(self, offset: isize) -> Self;
}

/// A generic Tensor representation.
#[derive(Debug, Clone)]
pub struct PtrTensor<P> {
    ptr: P,
    shape: StridedShape,
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
}

impl<P: OffsetPtr> PtrTensor<P> {
    fn offset(&self, offset: isize, shape: StridedShape) -> Self {
        Self::from_parts(self.ptr.clone().offset_bytes(4 * offset), shape)
    }

    pub fn permute(&self, permutation: &[usize]) -> Self {
        self.offset(0, self.shape.permute(permutation))
    }

    pub fn view(&self, new_shape: Vec<usize>) -> Self {
        self.offset(0, self.shape.view(new_shape.clone()).unwrap())
    }

    pub fn slice(&self, axis: usize, range: SliceRange) -> Self {
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

        self.slice(axis, SliceRange::simple(index, index + 1)).view(new_shape)
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
}
