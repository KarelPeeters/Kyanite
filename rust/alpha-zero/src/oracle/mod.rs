use std::fmt::Debug;

use board_game::board::{Board, Outcome};

pub mod syzygy;

#[derive(Debug)]
pub struct OracleEvaluation<B: Board> {
    pub best_outcome: Outcome,
    pub best_move: Option<B::Move>,
}

pub trait Oracle<B: Board>: Debug {
    /// Evaluate the given position, returning the best outcome and best move.
    /// Returns None if this oracle does not know about this position.
    fn evaluate(&self, board: &B) -> Option<OracleEvaluation<B>>;

    /// The same as [Oracle::evaluate] but may be faster since it doesn't need to find the best move.
    fn best_outcome(&self, board: &B) -> Option<Outcome> {
        self.evaluate(board).map(|e| e.best_outcome)
    }
}

/// An oracle without any knowledge, meaning that it only returns evaluations for terminal positions.
#[derive(Debug)]
pub struct DummyOracle;

impl<B: Board> Oracle<B> for DummyOracle {
    fn evaluate(&self, board: &B) -> Option<OracleEvaluation<B>> {
        board.outcome().map(|outcome| {
            OracleEvaluation { best_outcome: outcome, best_move: None }
        })
    }
}
