use board_game::board::Board;

use kz_core::network::ZeroEvaluation;

/// A full game.
#[derive(Debug)]
pub struct Simulation<B: Board> {
    pub positions: Vec<Position<B>>,
    // can be non-terminal if the game was stopped by the length limit
    pub final_board: B,
}

/// A single position in a game.
#[derive(Debug)]
pub struct Position<B: Board> {
    pub board: B,
    pub is_full_search: bool,
    pub played_mv: B::Move,

    pub zero_visits: u64,
    pub zero_evaluation: ZeroEvaluation<'static>,
    pub net_evaluation: ZeroEvaluation<'static>,
}

impl<B: Board> Simulation<B> {
    pub fn start_board(&self) -> &B {
        match self.positions.get(0) {
            Some(pos) => &pos.board,
            None => &self.final_board,
        }
    }
}
