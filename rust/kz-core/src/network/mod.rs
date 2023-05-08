use std::borrow::{Borrow, Cow};
use std::fmt::Debug;

use board_game::board::Board;

use crate::network::job_channel::JobClient;
use crate::zero::values::ZeroValuesPov;

pub mod common;
pub mod dummy;
pub mod muzero;
pub mod symmetry;

pub mod cpu;
pub mod cudnn;

pub mod multibatch;

pub mod job_channel;

/// A board evaluation, either as returned by the network or as the final output of a zero tree search.
#[derive(Debug, Clone)]
pub struct ZeroEvaluation<'a> {
    /// The (normalized) values.
    pub values: ZeroValuesPov,

    /// The (normalized) policy "vector", only containing the available moves in the order they are yielded by `available_moves`.
    pub policy: Cow<'a, [f32]>,
}

impl ZeroEvaluation<'_> {
    pub fn shallow_clone(&self) -> ZeroEvaluation {
        ZeroEvaluation {
            values: self.values,
            policy: Cow::Borrowed(self.policy.borrow()),
        }
    }
}

pub type EvalClient<B> = JobClient<B, ZeroEvaluation<'static>>;

pub trait Network<B: Board>: Debug {
    fn max_batch_size(&self) -> usize;

    // TODO change arg type to impl exact size iterator instead?
    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation<'static>>;

    fn evaluate(&mut self, board: &B) -> ZeroEvaluation<'static> {
        let mut result = self.evaluate_batch(&[board]);
        assert_eq!(result.len(), 1);
        result.pop().unwrap()
    }
}

impl<B: Board, N: Network<B>> Network<B> for &mut N {
    fn max_batch_size(&self) -> usize {
        (**self).max_batch_size()
    }

    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation<'static>> {
        (**self).evaluate_batch(boards)
    }
}
