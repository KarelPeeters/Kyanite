use board_game::board::{Board, Outcome};

use crate::network::ZeroEvaluation;

/// A full game.
#[derive(Debug)]
pub struct Simulation<B: Board> {
    pub outcome: Outcome,
    pub positions: Vec<Position<B>>,
}

/// A single position in a game.
#[derive(Debug)]
pub struct Position<B: Board> {
    pub board: B,
    pub should_store: bool,
    pub played_mv: B::Move,

    pub zero_visits: u64,
    pub zero_evaluation: ZeroEvaluation<'static>,
    pub net_evaluation: ZeroEvaluation<'static>,
}
