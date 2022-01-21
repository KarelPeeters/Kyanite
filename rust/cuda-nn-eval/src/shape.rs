use std::fmt::Debug;

use itertools::{Itertools, zip_eq};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StridedShape {
    shape: Vec<usize>,
    strides: Vec<usize>,
    has_simple_strides: bool,
}

impl StridedShape {
    pub fn new(shape: Vec<usize>, strides: Vec<usize>) -> Self {
        assert_eq!(shape.len(), strides.len(), "Shape and stride rank mismatch");
        let has_simple_strides = itertools::equal(strides.iter().copied(), simple_strides(&shape));
        StridedShape { shape, strides, has_simple_strides }
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

    pub fn slice(&self, axis: usize, start: usize, end: usize) -> StridedShape {
        // everything stays the same except the size of the sliced axis
        let mut new_shape = self.shape.clone();
        new_shape[axis] = end - start;

        let new_strides = self.strides.clone();
        StridedShape::new(new_shape, new_strides)
    }

    pub fn view(&self, new_shape: Vec<usize>) -> Option<StridedShape> {
        // implementation roughly based on pytorch computeStride_impl:
        // https://github.com/pytorch/pytorch/blob/560cd881956bbf425251d63f0ff0f9085a759447/aten/src/ATen/TensorUtils.cpp#L335-L346

        let new_size = new_shape.iter().copied().product();
        assert_eq!(self.size(), new_size, "Size cannot change during view, cannot go from {:?} to {:?}", self, new_shape);

        if self.size() == 0 || self.rank() == 0 {
            return Some(StridedShape::new_simple(new_shape));
        }

        let mut new_strides = vec![0; new_shape.len()];
        let mut next_d = 0;

        let mut failed = false;

        self.for_each_continuous_group(|group_size, group_stride| {
            if failed { return; };

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

fn visit_strided_indices_impl(start: usize, shape: &[usize], strides: &[usize], f: &mut impl FnMut(usize)) {
    match shape {
        [] => f(start),
        [size_curr, size_rest @ ..] => {
            for i in 0..*size_curr {
                visit_strided_indices_impl(
                    start + i * strides[0],
                    size_rest,
                    &strides[1..],
                    f,
                )
            }
        }
    }
}

#[cfg(test)]
mod test {
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
    fn rank_zero() {
        let shape = StridedShape::new(vec![], vec![]);
        assert_eq!(
            collect_groups(&shape),
            (vec![0], vec![1]),
        );
        assert_eq!(
            shape.view(vec![1, 1, 1]),
            Some(StridedShape::new(vec![1, 1, 1], vec![1, 1, 1])),
        );
    }

    #[test]
    fn size_zero() {
        let shape = StridedShape::new(vec![2, 3, 0, 5], vec![0, 0, 0, 2]);
        assert_eq!(
            collect_groups(&shape),
            (vec![0], vec![1])
        );
        assert_eq!(
            shape.view(vec![0]),
            Some(StridedShape::new(vec![0], vec![1])),
        );
        assert_eq!(
            shape.view(vec![12, 0]),
            Some(StridedShape::new(vec![12, 0], vec![0, 1])),
        );
    }

    #[test]
    fn simple() {
        let shape = StridedShape::new(vec![2, 3, 4, 3, 2], vec![72, 24, 6, 2, 1]);
        assert!(shape.has_simple_strides());
        assert_eq!(
            collect_groups(&shape),
            (vec![144], vec![1])
        );
        assert_eq!(
            shape.view(vec![144]),
            Some(StridedShape::new(vec![144], vec![1])),
        );
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
    fn split() {
        let shape = StridedShape::new(vec![2, 3, 4], vec![24, 8, 1]);
        assert_eq!(
            collect_groups(&shape),
            (vec![6, 4], vec![8, 1])
        );
        assert_eq!(
            shape.view(vec![6, 4]),
            Some(StridedShape::new(vec![6, 4], vec![8, 1])),
        );
        assert_eq!(
            shape.view(vec![24]),
            None,
        );
    }
}