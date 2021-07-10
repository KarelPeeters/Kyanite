use std::marker::PhantomData;

use crate::board::Board;
use crate::util::bot_game::Bot;

/// The value heuristic used in a minimax search.
///
/// * `value` returns the value of the given board from the next player POV.
///     `+inf` means forced win, `-inf` means forced loss.
/// * `value_delta` is used to speed up value calculations, it must satisfy:
///     `value(b) + value_delta(b, mv) = value(b.clone_and_move(mv))`
pub trait Heuristic<B: Board> {
    fn value(&self, board: &B) -> f32;

    fn value_delta(&self, board: &B, mv: B::Move) -> f32 {
        self.value(&board.clone_and_play(mv)) - self.value(board)
    }
}

pub struct MinimaxResult<B: Board> {
    pub best_move: B::Move,
    pub value: f32,
}

pub fn minimax<B: Board, H: Heuristic<B>>(board: &B, heuristic: &H, depth: u32) -> MinimaxResult<B> {
    assert!(depth > 0);
    assert!(!board.is_done());

    let (best_move, value) = negamax_recurse(heuristic, board, heuristic.value(board), depth, -f32::INFINITY, f32::INFINITY);
    MinimaxResult { best_move: best_move.unwrap(), value }
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
) -> (Option<B::Move>, f32) {
    if depth_left == 0 || board.is_done() {
        return (None, board_heuristic);
    }

    let mut best_value = -f32::INFINITY;
    let mut best_move: Option<B::Move> = None;
    let mut alpha = alpha;

    for mv in board.available_moves() {
        let child = board.clone_and_play(mv);
        let child_heuristic = -board_heuristic + heuristic.value_delta(board, mv);

        let child_value = -negamax_recurse(
            heuristic,
            &child,
            child_heuristic,
            depth_left - 1,
            -beta,
            -alpha,
        ).1;

        if child_value >= beta {
            return (None, child_value);
        }

        if child_value > best_value || best_move.is_none() {
            best_value = child_value;
            best_move = Some(mv);

            alpha = f32::max(alpha, child_value)
        }
    }

    (best_move, best_value)
}

pub struct MiniMaxBot<B: Board, H: Heuristic<B>> {
    depth: u32,
    heuristic: H,
    ph: PhantomData<B>,
}

impl<B: Board, H: Heuristic<B>> MiniMaxBot<B, H> {
    pub fn new(depth: u32, heuristic: H) -> Self {
        MiniMaxBot { depth, heuristic, ph: PhantomData }
    }
}

impl<B: Board, H: Heuristic<B>> Bot<B> for MiniMaxBot<B, H> {
    fn select_move(&mut self, board: &B) -> B::Move {
        minimax(board, &self.heuristic, self.depth).best_move
    }
}

