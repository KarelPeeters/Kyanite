use std::fmt::Display;

use itertools::zip;

/// Calculates `D_KDL(P || Q)`, read as _the divergence of P from Q_,
/// a measure of how different two probability distributions are.
/// See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence.
pub fn kdl_divergence(p: &[f32], q: &[f32]) -> f32 {
    assert_eq!(p.len(), q.len());

    zip(p, q).map(|(&p, &q)| p * (p / q).ln()).sum()
}

pub fn ceil_div<T: num_traits::PrimInt + Display>(x: T, y: T) -> T {
    let zero = T::zero();
    assert!(x >= zero && y > zero, "Invalid values for ceil_div({}, {})", x, y);
    (x + y - T::one()) / y
}
