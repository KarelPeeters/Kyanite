use std::cmp::Reverse;
use std::fmt::Debug;

use itertools::{zip, zip_eq, Itertools};

use cuda_sys::wrapper::descriptor::{FilterDescriptor, TensorDescriptor};
use nn_graph::graph::SliceRange;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StridedShape {
    shape: Vec<usize>,
    strides: Vec<usize>,
    has_simple_strides: bool,
    has_dense_strides: bool,
}

impl StridedShape {
    pub fn new(shape: Vec<usize>, strides: Vec<usize>) -> Self {
        assert_eq!(shape.len(), strides.len(), "Shape and stride rank mismatch");

        let has_simple_strides = &strides == &simple_strides(&shape);
        let has_dense_strides = has_dense_strides(&shape, &strides);

        StridedShape {
            shape,
            strides,
            has_simple_strides,
            has_dense_strides,
        }
    }

    pub fn new_simple(shape: Vec<usize>) -> Self {
        let strides = simple_strides(&shape);
        StridedShape::new(shape, strides)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn has_simple_strides(&self) -> bool {
        self.has_simple_strides
    }

    pub fn has_dense_strides(&self) -> bool {
        self.has_dense_strides
    }

    pub fn visit_strided_indices(&self, mut f: impl FnMut(usize)) {
        visit_strided_indices_impl(0, &self.shape, &self.strides, &mut f)
    }

    pub fn size(&self) -> usize {
        self.shape.iter().copied().product()
    }

    pub fn strided_size(&self) -> usize {
        self.max_index().map_or(0, |x| x + 1)
    }

    pub fn max_index(&self) -> Option<usize> {
        let mut total = 0;
        for (&size, &stride) in zip_eq(&self.shape, &self.strides) {
            if size == 0 {
                return None;
            } else {
                total += (size - 1) * stride;
            }
        }
        Some(total)
    }

    pub fn slice(&self, axis: usize, range: SliceRange) -> StridedShape {
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        range.assert_valid();
        let SliceRange { start, end, step } = range;

        new_shape[axis] = (end - start) / step;
        new_strides[axis] *= step;

        StridedShape::new(new_shape, new_strides)
    }

    pub fn view(&self, new_shape: Vec<usize>) -> Option<StridedShape> {
        // implementation roughly based on pytorch computeStride_impl:
        // https://github.com/pytorch/pytorch/blob/560cd881956bbf425251d63f0ff0f9085a759447/aten/src/ATen/TensorUtils.cpp#L335-L346

        let new_size = new_shape.iter().copied().product::<usize>();
        assert_eq!(
            self.size(),
            new_size,
            "Size cannot change during view, cannot go from {:?} to {:?}",
            self,
            new_shape
        );

        if self.size() == 0 || self.rank() == 0 {
            return Some(StridedShape::new_simple(new_shape));
        }

        let mut new_strides = vec![0; new_shape.len()];
        let mut next_d = 0;

        let mut failed = false;

        self.for_each_continuous_group(|group_size, group_stride| {
            if failed {
                return;
            };

            let mut left_group_size = group_size;
            while left_group_size > 1 {
                if left_group_size % new_shape[next_d] == 0 {
                    left_group_size /= new_shape[next_d];
                    new_strides[next_d] = left_group_size * group_stride;
                    next_d += 1;
                } else {
                    failed = true;
                    return;
                }
            }
        });

        if failed {
            None
        } else {
            // complete the strides for trailing 1-sized dims
            for d in next_d..new_shape.len() {
                assert_eq!(new_shape[d], 1);
                new_strides[d] = 1;
            }

            Some(StridedShape::new(new_shape, new_strides))
        }
    }

    fn for_each_continuous_group(&self, mut f: impl FnMut(usize, usize)) {
        if self.size() == 0 || self.rank() == 0 {
            f(0, 1);
            return;
        }

        let mut group_size = 1;
        let mut prev_stride = None;

        for (&d_size, &d_stride) in zip_eq(&self.shape, &self.strides) {
            if let Some(prev_stride) = prev_stride {
                if prev_stride != d_size * d_stride {
                    //finish previous group
                    f(group_size, prev_stride);
                    group_size = 1;
                }
            }

            group_size *= d_size;
            prev_stride = Some(d_stride)
        }

        if let Some(prev_stride) = prev_stride {
            //finish last group
            f(group_size, prev_stride)
        }
    }

    pub fn permute(&self, permutation: &[usize]) -> StridedShape {
        assert_eq!(permutation.len(), self.rank());
        assert!(permutation.iter().all_unique());

        // just permute the shape and strides
        let new_shape = permutation.iter().map(|&i| self.shape()[i]).collect();
        let new_strides = permutation.iter().map(|&i| self.strides()[i]).collect();

        StridedShape::new(new_shape, new_strides)
    }

    pub fn descriptor(&self) -> TensorDescriptor {
        let mut shape = self.shape.iter().map(|&x| x as i32).collect_vec();
        let mut strides = self.strides.iter().map(|&x| x as i32).collect_vec();

        // tensor descriptors and some cudnn operations seem to break with ranks < 4,
        //   so pad the rank until it's large enough
        while shape.len() < 4 {
            shape.push(1);
            strides.push(1);
        }

        TensorDescriptor::new(shape, strides)
    }

    pub fn filter_descriptor(&self) -> FilterDescriptor {
        assert_eq!(4, self.rank(), "Filter must have rank 4");
        assert!(self.has_simple_strides(), "Filter must have simple strides");

        let dims = self.shape();
        FilterDescriptor::new(dims[0] as i32, dims[1] as i32, dims[2] as i32, dims[3] as i32)
    }
}

fn simple_strides(shape: &[usize]) -> Vec<usize> {
    let mut result = vec![];
    let mut next_stride = 1;

    for &size in shape.iter().rev() {
        result.push(next_stride);
        next_stride *= size;
    }

    result.reverse();
    result
}

/// Whether the given shape covers every value within its data range.
/// This is equivalent to asking whether any possible permutation of the shape has simple strides.
fn has_dense_strides(shape: &[usize], strides: &[usize]) -> bool {
    assert_eq!(shape.len(), strides.len());

    let pairs = zip(shape.iter().copied(), strides.iter().copied())
        .sorted_by_key(|x| Reverse(x.1))
        .collect_vec();

    let sorted_shape = pairs.iter().map(|&x| x.0).collect_vec();
    let sorted_strides = pairs.iter().map(|&x| x.1).collect_vec();

    simple_strides(&sorted_shape) == sorted_strides
}

fn visit_strided_indices_impl(start: usize, shape: &[usize], strides: &[usize], f: &mut impl FnMut(usize)) {
    match shape {
        [] => f(start),
        [size_curr, size_rest @ ..] => {
            for i in 0..*size_curr {
                visit_strided_indices_impl(start + i * strides[0], size_rest, &strides[1..], f)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use nn_graph::graph::SliceRange;

    use crate::shape::StridedShape;

    fn collect_groups(shape: &StridedShape) -> (Vec<usize>, Vec<usize>) {
        let mut sizes = vec![];
        let mut strides = vec![];
        shape.for_each_continuous_group(|group_size, group_stride| {
            sizes.push(group_size);
            strides.push(group_stride);
        });
        (sizes, strides)
    }

    #[test]
    fn view_rank_zero() {
        let shape = StridedShape::new(vec![], vec![]);
        assert_eq!(collect_groups(&shape), (vec![0], vec![1]),);
        assert_eq!(
            shape.view(vec![1, 1, 1]),
            Some(StridedShape::new(vec![1, 1, 1], vec![1, 1, 1])),
        );
    }

    #[test]
    fn view_size_zero() {
        let shape = StridedShape::new(vec![2, 3, 0, 5], vec![0, 0, 0, 2]);
        assert_eq!(collect_groups(&shape), (vec![0], vec![1]));
        assert_eq!(shape.view(vec![0]), Some(StridedShape::new(vec![0], vec![1])),);
        assert_eq!(
            shape.view(vec![12, 0]),
            Some(StridedShape::new(vec![12, 0], vec![0, 1])),
        );
    }

    #[test]
    fn view_simple() {
        let shape = StridedShape::new(vec![2, 3, 4, 3, 2], vec![72, 24, 6, 2, 1]);
        assert!(shape.has_simple_strides());
        assert_eq!(collect_groups(&shape), (vec![144], vec![1]));
        assert_eq!(shape.view(vec![144]), Some(StridedShape::new(vec![144], vec![1])),);
        assert_eq!(
            shape.view(vec![72, 2]),
            Some(StridedShape::new(vec![72, 2], vec![2, 1])),
        );
        assert_eq!(
            shape.view(vec![72, 2, 1, 1, 1]),
            Some(StridedShape::new(vec![72, 2, 1, 1, 1], vec![2, 1, 1, 1, 1])),
        );
    }

    #[test]
    fn view_split() {
        let shape = StridedShape::new(vec![2, 3, 4], vec![24, 8, 1]);
        assert_eq!(collect_groups(&shape), (vec![6, 4], vec![8, 1]));
        assert_eq!(shape.view(vec![6, 4]), Some(StridedShape::new(vec![6, 4], vec![8, 1])),);
        assert_eq!(shape.view(vec![24]), None,);
    }

    #[test]
    fn slice_simple() {
        let shape = StridedShape::new(vec![2, 3, 4], vec![24, 8, 1]);
        assert_eq!(
            shape.slice(1, SliceRange::new(0, 4, 2)),
            StridedShape::new(vec![2, 2, 4], vec![24, 16, 1])
        )
    }
}
