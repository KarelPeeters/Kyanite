use internal_iterator::InternalIterator;

use crate::ai::Bot;
use crate::ai::minimax::{Heuristic, minimax};
use crate::board::{Board, Outcome, Player};

/// Heuristic with `bound()-length` for win, `-bound()+length` for loss and 0 for draw.
/// This means the sign of the final minimax value means forced win, forced loss or unknown, and the selected move is
/// the shortest win of the longest loss.
#[derive(Debug)]
pub struct SolverHeuristic;

impl<B: Board> Heuristic<B> for SolverHeuristic {
    type V = i32;

    fn bound(&self) -> Self::V {
        i32::MAX
    }

    fn value(&self, board: &B, length: u32) -> i32 {
        board.outcome().map_or(0, |p| {
            p.pov(board.next_player()).sign::<i32>() * (i32::MAX - length as i32)
        })
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

    //TODO this is kind of ugly, consider writing a function try_fold or something
    //  that handles this back and forth conversion
    let result = board.available_moves().find_map(|mv| {
        let child = board.clone_and_play(mv);

        match is_double_forced_draw(&child, depth - 1) {
            Ok(true) => None,
            Ok(false) => Some(false),
            Err(()) => Some(true),
        }
    });

    match result {
        Some(true) => Err(()),
        Some(false) => Ok(false),
        None => Ok(true)
    }
}

#[derive(Debug)]
pub struct SolverBot {
    depth: u32,
}

impl SolverBot {
    pub fn new(depth: u32) -> Self {
        assert!(depth > 0);
        SolverBot { depth }
    }
}

impl<B: Board> Bot<B> for SolverBot {
    fn select_move(&mut self, board: &B) -> B::Move {
        minimax(board, &SolverHeuristic, self.depth).best_move.unwrap()
    }
}