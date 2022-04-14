use std::borrow::{Borrow, Cow};

use crate::zero::node::ZeroValues;

pub mod node;
pub mod tree;

pub mod step;
pub mod wrapper;

/// A board evaluation, either as returned by the network or as the final output of a zero tree search.
#[derive(Debug, Clone)]
pub struct MuZeroEvaluation<'a> {
    /// The (normalized) values.
    pub values: ZeroValues,

    /// The (normalized) policy "vector", containing all possible moves.
    pub policy: Cow<'a, [f32]>,
}

impl MuZeroEvaluation<'_> {
    pub fn shallow_clone(&self) -> MuZeroEvaluation {
        MuZeroEvaluation {
            values: self.values,
            policy: Cow::Borrowed(self.policy.borrow()),
        }
    }

    pub fn assert_normalized_or_nan(&self) {
        let policy_sum = self.policy.iter().copied().sum::<f32>();
        if !policy_sum.is_nan() {
            assert!(
                (policy_sum - 1.0).abs() < 0.001,
                "Expected normalized policy, got {:?} with sum {}",
                self.policy,
                policy_sum
            );
        }

        let wdl_sum = self.values.wdl.sum();
        if !wdl_sum.is_nan() {
            assert!(
                (wdl_sum - 1.0).abs() < 0.001,
                "Expected normalized wdl, got {:?} with sum {}",
                wdl_sum,
                policy_sum
            );
        }
    }
}
