use std::fmt::Debug;

use rand::distributions::Distribution;
use rand::Rng;
use rand::seq::SliceRandom;

/// The symmetry group associated with a Board. An instance of this group maps a board and moves such that everything
/// about the board and its state is invariant under this mapping.
pub trait Symmetry: 'static + Debug + Copy + Clone + Eq + PartialEq {
    fn all() -> &'static [Self];
    fn identity() -> Self;
    fn inverse(self) -> Self;
}

struct SymmetryDistribution;

impl<S: Symmetry + Sized> Distribution<S> for SymmetryDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> S {
        *S::all().choose(rng)
            .expect("A symmetry group cannot be empty")
    }
}

/// The trivial symmetry group with only the identity, can be used as a conservative implementation.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct UnitSymmetry;

impl Symmetry for UnitSymmetry {
    fn all() -> &'static [Self] { &[Self] }
    fn identity() -> Self { Self }
    fn inverse(self) -> Self { Self }
}

/// The D4 symmetry group that can represent any combination of
/// flips, rotating and transposing, which result in 8 distinct elements.
///
/// The `Default::default()` value means no transformation.
///
/// The representation is such that first x and y are optionally transposed,
/// then each axis is optionally flipped separately.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct D4Symmetry {
    pub transpose: bool,
    pub flip_x: bool,
    pub flip_y: bool,
}

impl D4Symmetry {
    pub const fn new(transpose: bool, flip_x: bool, flip_y: bool) -> Self {
        D4Symmetry { transpose, flip_x, flip_y }
    }

    pub fn map_xy<V: Copy + std::ops::Sub<Output=V>>(self, mut x: V, mut y: V, max: V) -> (V, V) {
        if self.transpose { std::mem::swap(&mut x, &mut y) };
        if self.flip_x { x = max - x };
        if self.flip_y { y = max - y };
        (x, y)
    }
}

impl Symmetry for D4Symmetry {
    fn all() -> &'static [Self] {
        const ALL: [D4Symmetry; 8] = [
            D4Symmetry::new(false, false, false),
            D4Symmetry::new(false, false, true),
            D4Symmetry::new(false, true, false),
            D4Symmetry::new(false, true, true),
            D4Symmetry::new(true, false, false),
            D4Symmetry::new(true, false, true),
            D4Symmetry::new(true, true, false),
            D4Symmetry::new(true, true, true),
        ];
        &ALL
    }

    fn identity() -> Self {
        D4Symmetry::new(false, false, false)
    }

    fn inverse(self) -> Self {
        D4Symmetry::new(
            self.transpose,
            if self.transpose { self.flip_y } else { self.flip_x },
            if self.transpose { self.flip_x } else { self.flip_y },
        )
    }
}
