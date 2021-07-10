use std::marker::PhantomData;

use crate::ai::Bot;
use crate::board::Board;
use std::ops::Neg;
use std::cmp::max;

pub trait Heuristic<B: Board> {
    /// The type used to represent the heuristic value of a board.
    type V: Copy + Ord + Neg<Output=Self::V>;

    /// Return a value V that such that for any possible value `v`: `-bound <= v <= bound`.
    fn bound(&self) -> Self::V;

    /// Return the heuristic value for the given board from the the next player POV.
    /// This value must induce a zero-sum game.
    fn value(&self, board: &B) -> Self::V;

    /// Return the value of `child`, given the previous board, its value and the move that was just played.
    /// This function can be overridden to improve performance.
    ///
    /// Given:
    /// * `board.clone_and_play(mv) == child`
    /// * `value(board) == board_value`
    ///
    /// This function must ensure that
    /// * `value(child) == value_update(board, board_value, mv, child)`
    #[allow(unused_variables)]
    fn value_update(&self, board: &B, board_value: Self::V, mv: B::Move, child: &B) -> Self::V {
        self.value(child)
    }
}

pub struct MinimaxResult<V, M> {
    /// The value of this board.
    pub value: V,

    /// The best move to play, `None` is the board is done or the search depth was 0
    pub best_move: Option<M>,
}

/// Evaluate the board using minimax with the given heuristic up to the given depth.
pub fn minimax<B: Board, H: Heuristic<B>>(board: &B, heuristic: &H, depth: u32) -> MinimaxResult<H::V, B::Move> {
    let result = negamax_recurse(
        heuristic,
        board,
        heuristic.value(board),
        depth,
        -heuristic.bound(),
        heuristic.bound()
    );

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
    board_heuristic: H::V,
    depth_left: u32,
    alpha: H::V,
    beta: H::V,
) -> MinimaxResult<H::V, B::Move> {
    if depth_left == 0 || board.is_done() {
        return MinimaxResult { value: board_heuristic, best_move: None };
    }

    let mut best_value = -heuristic.bound();
    let mut best_move: Option<B::Move> = None;
    let mut alpha = alpha;

    for mv in board.available_moves() {
        let child = board.clone_and_play(mv);
        let child_heuristic = heuristic.value_update(board, board_heuristic, mv, &child);

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

            alpha = max(alpha, child_value)
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

