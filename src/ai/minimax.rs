use std::marker::PhantomData;

use crate::ai::Bot;
use crate::board::Board;

/// The value heuristic used in a minimax search.
///
/// * `value` returns the value of the given board from the next player POV.
///      This value must induce a zero-sum game, if the players are flipped the value should be negated.
/// * `value_update` can be overridden to incrementally update the value to speed up calculations.
pub trait Heuristic<B: Board> {
    fn value(&self, board: &B) -> f32;

    /// Return the value of `child`, given the previous board, its value and the move that was just played.
    /// Given:
    /// * `board.clone_and_play(mv) == child`
    /// * `value(board) == board_value`
    ///
    /// This function must ensure that
    /// * `value(child) == value_update(board, board_value, mv, child)`
    #[allow(unused_variables)]
    fn value_update(&self, board: &B, board_value: f32, mv: B::Move, child: &B) -> f32 {
        self.value(child)
    }
}

pub struct MinimaxResult<B: Board> {
    pub value: f32,

    /// The best move to play, `None` is the board is done or the search depth was 0
    pub best_move: Option<B::Move>,
}

/// Evaluate the board using minimax with the given heuristic up to the given depth.
pub fn minimax<B: Board, H: Heuristic<B>>(board: &B, heuristic: &H, depth: u32) -> MinimaxResult<B> {
    let board_value = heuristic.value(board);
    assert!(!board_value.is_nan());

    let result = negamax_recurse(heuristic, board, board_value, depth, -f32::INFINITY, f32::INFINITY);

    if result.best_move.is_none() {
        assert!(board.is_done() || depth == 0, "Implementation error in negamax");
    }

    result
}

/// Fail-Soft Alpha-Beta Negamax, implementation based on
/// https://www.chessprogramming.org/Alpha-Beta#cite_note-9
fn negamax_recurse<B: Board, H: Heuristic<B>>(
    heuristic: &H,
    board: &B,
    board_heuristic: f32,
    depth_left: u32,
    alpha: f32,
    beta: f32,
) -> MinimaxResult<B> {
    if depth_left == 0 || board.is_done() {
        return MinimaxResult { value: board_heuristic, best_move: None };
    }

    let mut best_value = -f32::INFINITY;
    let mut best_move: Option<B::Move> = None;
    let mut alpha = alpha;

    for mv in board.available_moves() {
        let child = board.clone_and_play(mv);
        let child_heuristic = heuristic.value_update(board, board_heuristic, mv, &child);
        assert!(!child_heuristic.is_nan());

        let child_value = -negamax_recurse(
            heuristic,
            &child,
            child_heuristic,
            depth_left - 1,
            -beta,
            -alpha,
        ).value;

        if child_value >= beta {
            return MinimaxResult { value: child_value, best_move: Some(mv) };
        }

        if child_value > best_value || best_move.is_none() {
            best_value = child_value;
            best_move = Some(mv);

            alpha = f32::max(alpha, child_value)
        }
    }

    MinimaxResult { value: best_value, best_move: Some(best_move.unwrap()) }
}

pub struct MiniMaxBot<B: Board, H: Heuristic<B>> {
    depth: u32,
    heuristic: H,
    ph: PhantomData<B>,
}

impl<B: Board, H: Heuristic<B>> MiniMaxBot<B, H> {
    pub fn new(depth: u32, heuristic: H) -> Self {
        assert!(depth > 0, "requires depth>0 to find the best move");
        MiniMaxBot { depth, heuristic, ph: PhantomData }
    }
}

impl<B: Board, H: Heuristic<B>> Bot<B> for MiniMaxBot<B, H> {
    fn select_move(&mut self, board: &B) -> B::Move {
        assert!(!board.is_done());
        minimax(board, &self.heuristic, self.depth).best_move.unwrap()
    }
}

