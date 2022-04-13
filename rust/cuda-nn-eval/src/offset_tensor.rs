use crate::shape::StridedShape;
use nn_graph::graph::SliceRange;

/// A generic Tensor representation.
#[derive(Debug, Clone)]
pub struct OffsetTensor<I> {
    inner: I,
    offset: isize,
    shape: StridedShape,
}

impl<I> OffsetTensor<I> {
    pub fn from_parts(inner: I, offset: isize, shape: StridedShape) -> Self {
        OffsetTensor { inner, offset, shape }
    }

    pub fn inner(&self) -> &I {
        &self.inner
    }

    pub fn offset(&self) -> isize {
        self.offset
    }

    pub fn shape(&self) -> &StridedShape {
        &self.shape
    }
}

impl<I: Clone> OffsetTensor<I> {
    //TODO find a better name for this
    fn delta(&self, offset: isize, shape: StridedShape) -> Self {
        Self::from_parts(self.inner.clone(), self.offset + offset, shape)
    }

    pub fn permute(&self, permutation: &[usize]) -> Self {
        self.delta(0, self.shape.permute(permutation))
    }

    pub fn view(&self, new_shape: Vec<usize>) -> Self {
        self.delta(0, self.shape.view(new_shape.clone()).unwrap())
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

        self.delta(offset, result_shape)
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

        self.delta(offset, result_shape)
    }
}
