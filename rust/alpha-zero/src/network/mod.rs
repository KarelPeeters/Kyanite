use std::fmt::Debug;

use board_game::board::Board;

use board_game::wdl::WDL;

pub mod dummy;

#[cfg(feature = "tch")]
pub mod torch_utils;

/// A board evaluation, either as returned by the network or as the final output of a zero tree search.
#[derive(Debug, Clone)]
pub struct ZeroEvaluation {
    /// The win, draw and loss probabilities, after normalization.
    pub wdl: WDL<f32>,

    /// The policy "vector", only containing the available moves in the order they are yielded by `available_moves`.
    pub policy: Vec<f32>,
}

pub trait Network<B: Board>: Debug {
    fn evaluate_batch(&mut self, boards: &[B]) -> Vec<ZeroEvaluation>;

    fn evaluate(&mut self, board: &B) -> ZeroEvaluation {
        let mut result = self.evaluate_batch(&[board.clone()]);
        assert_eq!(result.len(), 1);
        result.pop().unwrap()
    }
}

#[allow(dead_code)]
pub fn softmax(slice: &mut [f32]) {
    let mut sum = 0.0;
    for v in slice.iter_mut() {
        *v = v.exp();
        sum += *v;
    }
    for v in slice.iter_mut() {
        *v /= sum;
    }
}
