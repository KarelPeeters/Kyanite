use crate::ai::minimax::{Heuristic, minimax};
use crate::board::{Board, Player};

struct SolverHeuristic;

impl<B: Board> Heuristic<B> for SolverHeuristic {
    fn value(&self, board: &B) -> f32 {
        board.outcome().map_or(0.0, |p| p.sign(board.next_player()))
    }
}

/// Return which player can force a win if any. Both forced draws and unknown results are returned as `None`.
pub fn find_forcing_winner(board: &impl Board, depth: u32) -> Option<Player> {
    let value = minimax(board, &SolverHeuristic, depth).value;
    if value == 1.0 {
        Some(board.next_player())
    } else if value == -1.0 {
        Some(board.next_player().other())
    } else if value == 0.0 {
        None
    } else {
        panic!("Unexpected value {}", value)
    }
}