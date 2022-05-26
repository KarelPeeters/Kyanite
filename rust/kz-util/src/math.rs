use itertools::zip;

/// Calculates `D_KDL(P || Q)`, read as _the divergence of P from Q_,
/// a measure of how different two probability distributions are.
/// See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence.
pub fn kdl_divergence(p: &[f32], q: &[f32]) -> f32 {
    assert_eq!(p.len(), q.len());

    zip(p, q).map(|(&p, &q)| p * (p / q).ln()).sum()
}
