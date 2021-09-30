use board_game::board::Outcome;

use crate::network::ZeroEvaluation;

/// A full game.
#[derive(Debug)]
pub struct Simulation<B> {
    pub outcome: Outcome,
    pub positions: Vec<Position<B>>,
}

/// A single position in a game.
#[derive(Debug)]
pub struct Position<B> {
    pub board: B,
    pub should_store: bool,

    pub zero_visits: u64,
    pub zero_evaluation: ZeroEvaluation,
    pub net_evaluation: ZeroEvaluation,
}
