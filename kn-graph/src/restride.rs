use itertools::{Itertools, zip_eq};

use crate::graph::SliceRange;
use crate::onnx::typed_value::SignedSize;
use crate::shape;
use crate::shape::{Shape, Size};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Restride {
    pub old_shape: Shape,
    pub new_shape: Shape,

    pub offset: SignedSize,
    pub strides: Vec<SignedSize>,
}

impl Restride {
    pub fn assert_valid(&self) {
        // todo in-bounds (both offset and strides * shape)
        // todo stride and shape len the same
        // todo output shape match
        // todo size zero: all strides one
    }

    pub fn is_identity(&self) -> bool {
        self.assert_valid();
        let Restride { old_shape, new_shape, offset, strides } = self;
        offset.is_zero() && old_shape == new_shape && strides == &simple_strides(old_shape)
    }

    // TODO check that this fixes:
    //    consecutive views
    //    trivial broadcast
    //    trivial slice
    //    see if index=slice+view is merged nicely
    pub fn combine(first: &Restride, second: &Restride) -> Option<Restride> {
        todo!()
    }

    pub fn view(old_shape: Shape, new_shape: Shape) -> Restride {
        assert_eq!(
            old_shape.size(),
            new_shape.size(),
            "New shape {:?} must have the same size as old shape {:?}",
            new_shape,
            old_shape,
        );

        let strides = simple_strides(&new_shape);
        Restride {
            old_shape,
            new_shape,
            offset: SignedSize::ZERO,
            strides,
        }
    }

    pub fn broadcast(old_shape: Shape, new_shape: Shape) -> Restride {
        assert_eq!(
            old_shape.rank(),
            new_shape.rank(),
            "New shape {:?} must have the same rank as old shape {:?}",
            new_shape,
            old_shape,
        );

        // check that broadcasting is valid)
        for (&v, &n) in zip_eq(&new_shape.dims, &new_shape.dims) {
            assert!(
                v == n || v == Size::ONE,
                "Cannot broadcast from {:?} to {:?} because of axis ({}, {})",
                old_shape,
                new_shape,
                v,
                n
            );
        }

        todo!()
    }

    pub fn permute(old_shape: Shape, permutation: Vec<usize>) -> Restride {
        assert_eq!(
            permutation.len(),
            old_shape.rank(),
            "Permutation rank must match input shape, got {:?} and {:?}",
            permutation,
            old_shape
        );
        assert!(
            permutation.iter().all_unique(),
            "Permutation cannot contain repeated axis, got {:?}",
            permutation
        );
        assert!(
            permutation.iter().all(|&i| i < old_shape.rank()),
            "Permutation axis out of bounds, got {:?}",
            permutation
        );

        let old_strides = simple_strides(&old_shape);
        let new_dims = permutation.iter().map(|&i| old_shape[i]).collect_vec();
        let new_strides = permutation.iter().map(|&i| old_strides[i]).collect_vec();

        let new_shape = Shape::new(new_dims);

        Restride {
            old_shape,
            new_shape,
            offset: SignedSize::ZERO,
            strides: new_strides,
        }
    }

    pub fn slice(old_shape: Shape, axis: usize, range: SliceRange) -> Restride {
        old_shape.assert_has_axis(axis);

        // shape
        let old_size = old_shape.dims[axis].unwrap_fixed("Slice axis length");
        range.assert_in_bounds(old_size);
        let new_size = (range.end - range.start) / range.step;
        let new_shape = old_shape.clone().replace(axis, shape![new_size]);

        // strides
        let mut strides = simple_strides(&old_shape);
        let offset = SignedSize::from(Size::fixed(range.start)) * strides[axis];
        let step_size = SignedSize::from(Size::fixed(range.step));
        strides[axis] = strides[axis] * step_size;

        Restride {
            old_shape,
            new_shape,
            offset,
            strides,
        }
    }

    pub fn flip(old_shape: Shape, axis: usize) -> Restride {
        old_shape.assert_has_axis(axis);
        let axis_last = (old_shape[axis] - Size::ONE).expect("Cannot flip batch size axis");

        let mut strides = simple_strides(&old_shape);
        let offset = SignedSize::from(axis_last) * strides[axis];
        strides[axis] = -strides[axis];

        Restride {
            old_shape: old_shape.clone(),
            new_shape: old_shape,
            offset,
            strides,
        }
    }

    pub fn is_view(&self) -> bool {
        todo!()
    }

    pub fn is_broadcast(&self) -> bool {
        todo!()
    }
}

fn simple_strides(shape: &Shape) -> Vec<SignedSize> {
    let mut strides = Vec::with_capacity(shape.rank());
    let mut curr = SignedSize::ONE;
    for i in (0..shape.rank()).rev() {
        strides.push(curr);
        curr = curr * SignedSize::from(shape[i]);
    }
    strides.reverse();
    strides
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RestrideSlice {
    axis: usize,
    range: SliceRange,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RestrideFlip {
    axis: usize,
}

#[cfg(test)]
mod tests {
    #[test]
    fn trivial_restride() {
        // TODO
    }
}