use rand::Rng;

use crate::board::Board;

pub trait Bot<B: Board> {
    fn select_move(&mut self, board: &B) -> B::Move;
}

impl<B: Board, F: FnMut(&B) -> B::Move> Bot<B> for F {
    fn select_move(&mut self, board: &B) -> B::Move {
        self(board)
    }
}

pub struct RandomBot<R: Rng> {
    rng: R,
}

impl<B: Board, R: Rng> Bot<B> for RandomBot<R> {
    fn select_move(&mut self, board: &B) -> B::Move {
        board.random_available_move(&mut self.rng)
    }
}
