use std::convert::TryInto;
use std::fmt::{Debug, Display, Formatter};
use std::ops::ControlFlow;

use itertools::Itertools;

#[macro_export]
macro_rules! shape {
    [$($(*)? $value:expr),* $(,)?] => {
        $crate::shape::Shape::new(vec![$($crate::shape::Size::from($value)),*])
    };
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct Shape {
    pub dims: Vec<Size>,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Size {
    batch_exp: u32,
    fixed_factor: usize,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ConcreteShape {
    pub dims: Vec<usize>,
}

// TODO unify all shape types into a generic "Shape<S>"
//  what about strides?
impl Shape {
    pub const SCALAR: Shape = Shape { dims: vec![] };

    pub fn new(dims: Vec<Size>) -> Shape {
        Shape { dims }
    }

    pub fn single(size: Size) -> Shape {
        Shape { dims: vec![size] }
    }

    pub fn fixed(dims: &[usize]) -> Shape {
        let dims = dims.iter().map(|&d| Size::fixed(d)).collect_vec();
        Shape { dims }
    }

    pub fn ones(rank: usize) -> Shape {
        Shape::new(vec![Size::ONE; rank])
    }

    pub fn zeros(rank: usize) -> Shape {
        Shape::new(vec![Size::ZERO; rank])
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn assert_has_axis(&self, axis: usize) {
        assert!(axis < self.rank(), "Axis {} out of bounds for {:?}", axis, self);
    }

    pub fn as_fixed(&self) -> Option<ConcreteShape> {
        self.dims
            .iter()
            .map(|d| d.try_unwrap_fixed().ok_or(()))
            .try_collect()
            .ok()
            .map(|dims| ConcreteShape::new(dims))
    }

    pub fn unwrap_fixed(&self, what: &str) -> ConcreteShape {
        let dims = self.dims.iter().map(|d| d.unwrap_fixed(what)).collect_vec();
        ConcreteShape { dims }
    }

    pub fn eval(&self, batch_size: usize) -> ConcreteShape {
        let dims = self.dims.iter().map(|d| d.eval(batch_size)).collect_vec();
        ConcreteShape { dims }
    }

    pub fn size(&self) -> Size {
        self.dims.iter().copied().product()
    }

    pub fn unwrap_1(&self) -> Size {
        assert_eq!(1, self.dims.len(), "Expected rank 1 shape");
        self.dims[0]
    }

    pub fn unwrap_2(&self) -> [Size; 2] {
        self.dims
            .as_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("Expected rank 2 shape, got {:?}", self))
    }

    pub fn unwrap_3(&self) -> [Size; 3] {
        self.dims
            .as_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("Expected rank 3 shape, got {:?}", self))
    }

    pub fn unwrap_4(&self) -> [Size; 4] {
        self.dims
            .as_slice()
            .try_into()
            .unwrap_or_else(|_| panic!("Expected rank 4 shape, got {:?}", self))
    }

    pub fn concat(mut self, other: &Shape) -> Shape {
        self.dims.extend_from_slice(&other.dims);
        self
    }

    pub fn batched(&self) -> Shape {
        shape![Size::BATCH].concat(self)
    }

    /// Build a new shape with the shape at `axis` replaced by `replacement`, the rest are kept as-is.
    pub fn replace(&self, axis: usize, replacement: Shape) -> Shape {
        self.replace_all(&[axis], replacement)
    }

    pub fn replace_all(&self, axes: &[usize], replacement: Shape) -> Shape {
        // validate axes
        assert!(axes.iter().all_unique(), "Axes must be unique, got {:?}", axes);

        for &axis in axes {
            self.assert_has_axis(axis);
        }

        // construct new shape
        let mut dims = vec![];
        for i in 0..self.rank() {
            if axes.contains(&i) {
                dims.extend_from_slice(&replacement.dims);
            } else {
                dims.push(self[i])
            }
        }

        Shape::new(dims)
    }

    /// Build a new shape with the shape at `axis` kept and all other axes replaced by `rest`.
    pub fn keep(&self, axis: usize, rest: Size) -> Shape {
        self.assert_has_axis(axis);

        let mut dims = self.dims.clone();
        for i in 0..self.rank() {
            if i != axis {
                dims[i] = rest;
            }
        }
        Shape::new(dims)
    }

    pub fn repeat_unary(&self, axis: usize, new_size: Size) -> Shape {
        self.assert_has_axis(axis);

        assert_eq!(
            self.dims[axis],
            Size::ONE,
            "Repeated axis {} must have length 1 for {:?}",
            axis,
            self
        );

        let mut dims = self.dims.clone();
        dims[axis] = new_size;
        Shape::new(dims)
    }

    pub fn insert(&self, axis: usize, size: Size) -> Shape {
        assert!(
            axis <= self.rank(),
            "Axis {} out of bounds for inserting into {:?}",
            axis,
            self
        );

        let mut dims = self.dims.clone();
        dims.insert(axis, size);
        Shape::new(dims)
    }

    pub fn split(&self, index: usize) -> (Shape, Shape) {
        assert!(
            index <= self.rank(),
            "Split index {} out of bounds for {:?}",
            index,
            self
        );

        let body = self.dims[..index].to_vec();
        let tail = self.dims[index..].to_vec();

        (Shape::new(body), Shape::new(tail))
    }
}

impl From<usize> for Size {
    fn from(fixed_factor: usize) -> Self {
        Size::fixed(fixed_factor)
    }
}

impl Size {
    pub const ZERO: Size = Size::new(0, 0);
    pub const ONE: Size = Size::new(0, 1);
    pub const BATCH: Size = Size::new(1, 1);

    pub const fn new(batch_exp: u32, fixed_factor: usize) -> Size {
        if fixed_factor == 0 {
            Size {
                batch_exp: 0,
                fixed_factor: 0,
            }
        } else {
            Size {
                batch_exp,
                fixed_factor,
            }
        }
    }

    pub const fn fixed(size: usize) -> Size {
        Size {
            batch_exp: 0,
            fixed_factor: size,
        }
    }

    pub const fn is_zero(&self) -> bool {
        matches!(
            self,
            Size {
                batch_exp: 0,
                fixed_factor: 0
            }
        )
    }

    pub fn eval(self, batch_size: usize) -> usize {
        batch_size.pow(self.batch_exp) * self.fixed_factor
    }

    pub fn try_unwrap_fixed(self) -> Option<usize> {
        if self.batch_exp == 0 {
            Some(self.fixed_factor)
        } else {
            None
        }
    }

    #[track_caller]
    pub fn unwrap_fixed(self, what: &str) -> usize {
        assert_eq!(0, self.batch_exp, "{} must be fixed, but got size {:?}", what, self);
        self.fixed_factor
    }

    pub fn floor_div(self, rhs: Self) -> Option<Self> {
        if self.batch_exp < rhs.batch_exp {
            None
        } else {
            Some(Size::new(
                self.batch_exp - rhs.batch_exp,
                self.fixed_factor / rhs.fixed_factor,
            ))
        }
    }
}

impl ConcreteShape {
    pub fn new(dims: Vec<usize>) -> Self {
        ConcreteShape { dims }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn unwrap_2(&self) -> [usize; 2] {
        self.dims.as_slice().try_into().expect("Expected rank 2 shape")
    }

    pub fn unwrap_3(&self) -> [usize; 3] {
        self.dims.as_slice().try_into().expect("Expected rank 2 shape")
    }

    pub fn unwrap_4(&self) -> [usize; 4] {
        self.dims.as_slice().try_into().expect("Expected rank 4 shape")
    }
}

impl std::ops::Add for Size {
    type Output = Option<Size>;

    fn add(self, rhs: Self) -> Self::Output {
        if self == Size::ZERO {
            return Some(rhs);
        }
        if rhs == Size::ZERO {
            return Some(self);
        }
        if self.batch_exp != rhs.batch_exp {
            return None;
        }

        Some(Size::new(self.batch_exp, self.fixed_factor + rhs.fixed_factor))
    }
}

impl std::ops::Sub for Size {
    type Output = Option<Size>;

    fn sub(self, rhs: Self) -> Self::Output {
        if rhs == Size::ZERO {
            return Some(self);
        }

        if self.batch_exp != rhs.batch_exp || self.fixed_factor < rhs.fixed_factor {
            return None;
        }

        Some(Size::new(self.batch_exp, self.fixed_factor - rhs.fixed_factor))
    }
}

impl std::ops::Mul for Size {
    type Output = Size;

    fn mul(self, rhs: Self) -> Self::Output {
        Size::new(self.batch_exp + rhs.batch_exp, self.fixed_factor * rhs.fixed_factor)
    }
}

impl std::ops::Div for Size {
    type Output = Option<Size>;

    fn div(self, rhs: Self) -> Self::Output {
        if self.batch_exp < rhs.batch_exp || self.fixed_factor % rhs.fixed_factor != 0 {
            None
        } else {
            Some(Size::new(
                self.batch_exp - rhs.batch_exp,
                self.fixed_factor / rhs.fixed_factor,
            ))
        }
    }
}

impl std::iter::Sum<Size> for Option<Size> {
    fn sum<I: Iterator<Item = Size>>(mut iter: I) -> Self {
        let result = iter.try_fold(Size::ZERO, |a, s| match a + s {
            Some(v) => ControlFlow::Continue(v),
            None => ControlFlow::Break(()),
        });

        match result {
            ControlFlow::Continue(v) => Some(v),
            ControlFlow::Break(()) => None,
        }
    }
}

impl std::iter::Product for Size {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Size::fixed(1), |a, s| a * s)
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = Size;

    fn index(&self, axis: usize) -> &Self::Output {
        self.assert_has_axis(axis);
        &self.dims[axis]
    }
}

impl Debug for Shape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shape{}", self)
    }
}

impl Debug for Size {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Size({})", self)
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        fmt_shape_impl(f, &self.dims)
    }
}

impl Display for Size {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match (self.fixed_factor, self.batch_exp) {
            (a, 0) => write!(f, "{}", a),
            (1, 1) => write!(f, "B"),
            (a, 1) => write!(f, "{}B", a),
            (1, b) => write!(f, "B^{}", b),
            (a, b) => write!(f, "{}B^{}", a, b),
        }
    }
}

impl Display for ConcreteShape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        fmt_shape_impl(f, &self.dims)
    }
}

fn fmt_shape_impl(f: &mut Formatter, dims: &[impl Display]) -> Result<(), std::fmt::Error> {
    write!(f, "(")?;
    for i in 0..dims.len() {
        if i != 0 {
            write!(f, " x ")?;
        }

        write!(f, "{}", dims[i])?;
    }
    write!(f, ")")?;
    Ok(())
}
