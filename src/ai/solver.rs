use crate::ai::minimax::{Heuristic, minimax};
use crate::board::{Board, Outcome, Player};

struct SolverHeuristic;

impl<B: Board> Heuristic<B> for SolverHeuristic {
    type V = i8;

    fn bound(&self) -> Self::V {
        i8::MAX
    }

    fn value(&self, board: &B) -> i8 {
        board.outcome().map_or(0, |p| p.sign(board.next_player()))
    }
}

/// Return which player can force a win if any. Both forced draws and unknown results are returned as `None`.
pub fn find_forcing_winner(board: &impl Board, depth: u32) -> Option<Player> {
    let value = minimax(board, &SolverHeuristic, depth).value;
    if value == 1 {
        Some(board.next_player())
    } else if value == -1 {
        Some(board.next_player().other())
    } else if value == 0 {
        None
    } else {
        panic!("Unexpected value {}", value)
    }
}

/// Return whether this board is a double forced draw, ie. no matter what either player does the game can only end in a draw.
/// Returns `None` if the result is unknown.
pub fn is_double_forced_draw(board: &impl Board, depth: u32) -> Result<bool, ()> {
    if board.outcome() == Some(Outcome::Draw) { return Ok(true); }
    if board.outcome().is_some() { return Ok(false); }
    if depth == 0 { return Err(()); }

    for mv in board.available_moves() {
        let child = board.clone_and_play(mv);
        if !is_double_forced_draw(&child, depth - 1)? {
            return Ok(false);
        }
    }

    Ok(true)
}
