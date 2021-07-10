use crate::board::Board;

pub mod simple;
pub mod minimax;
pub mod solver;

pub trait Bot<B: Board> {
    fn select_move(&mut self, board: &B) -> B::Move;
}

impl<B: Board, F: FnMut(&B) -> B::Move> Bot<B> for F {
    fn select_move(&mut self, board: &B) -> B::Move {
        self(board)
    }
}