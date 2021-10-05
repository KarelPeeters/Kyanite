use std::convert::TryInto;
use std::fmt::{Debug, Display, Formatter};

use itertools::Itertools;

#[derive(Clone, Eq, PartialEq)]
pub struct Shape {
    pub dims: Vec<Size>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Size {
    batch_exp: u32,
    fixed_factor: usize,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ConcreteShape {
    pub dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<Size>) -> Shape {
        Shape { dims }
    }

    pub fn fixed(dims: &[usize]) -> Shape {
        let dims = dims.iter().map(|&d| Size::fixed(d)).collect_vec();
        Shape { dims }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn unwrap_fixed(&self) -> ConcreteShape {
        let dims = self.dims.iter().map(|d| d.unwrap_fixed()).collect_vec();
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
        self.dims.as_slice().try_into().expect("Expected rank 2 shape")
    }

    /// Returns a new shape with the same rank with all sizes set to 1, except the size at `index` is kept.
    pub fn all_ones_except(&self, index: usize) -> Shape {
        assert!(index < self.rank());

        let dims = self.dims.iter().enumerate()
            .map(|(i, &s)| {
                if i == index { s } else { Size::ONE }
            })
            .collect_vec();
        Shape { dims }
    }
}

impl Size {
    pub const ZERO: Size = Size { batch_exp: 0, fixed_factor: 0 };
    pub const ONE: Size = Size { batch_exp: 0, fixed_factor: 1 };
    pub const BATCH: Size = Size { batch_exp: 1, fixed_factor: 1 };

    pub fn new(batch_exp: u32, fixed_factor: usize) -> Size {
        Size { batch_exp, fixed_factor }
    }

    pub fn fixed(size: usize) -> Size {
        Size { batch_exp: 0, fixed_factor: size }
    }

    pub fn eval(self, batch_size: usize) -> usize {
        batch_size.pow(self.batch_exp) * self.fixed_factor
    }

    pub fn unwrap_fixed(self) -> usize {
        assert_eq!(0, self.batch_exp, "Expected fixed size, got {:?}", self);
        self.fixed_factor
    }

    pub fn unwrap_fixed_mut(&mut self) -> &mut usize {
        assert_eq!(0, self.batch_exp, "Expected fixed size, got {:?}", self);
        &mut self.fixed_factor
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

    pub fn unwrap_4(&self) -> [usize; 4] {
        self.dims.as_slice().try_into().expect("Expected rank 4 shape")
    }
}

impl std::ops::Mul for Size {
    type Output = Size;

    fn mul(self, rhs: Self) -> Self::Output {
        Size {
            batch_exp: self.batch_exp + rhs.batch_exp,
            fixed_factor: self.fixed_factor * rhs.fixed_factor,
        }
    }
}

impl std::ops::Div for Size {
    type Output = Size;

    fn div(self, rhs: Self) -> Self::Output {
        assert!(self.batch_exp >= rhs.batch_exp, "Not enough batch_exp remaining");
        assert_eq!(self.fixed_factor % rhs.fixed_factor, 0, "Fixed factor does not divide size");

        Size {
            batch_exp: self.batch_exp - rhs.batch_exp,
            fixed_factor: self.fixed_factor / rhs.fixed_factor,
        }
    }
}

impl std::iter::Product for Size {
    fn product<I: Iterator<Item=Self>>(iter: I) -> Self {
        iter.fold(Size::fixed(1), |a, s| a * s)
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = Size;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dims[index]
    }
}

impl std::ops::IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.dims[index]
    }
}

impl Debug for Shape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shape(")?;
        for i in 0..self.rank() {
            if i != 0 {
                write!(f, " x ")?;
            }

            write!(f, "{}", self.dims[i])?;
        }
        write!(f, ")")?;
        Ok(())
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