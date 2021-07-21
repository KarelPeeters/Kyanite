use std::cmp::max;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Neg;

use internal_iterator::InternalIterator;
use rand::Rng;

use crate::ai::Bot;
use crate::board::Board;

pub trait Heuristic<B: Board>: Debug {
    /// The type used to represent the heuristic value of a board.
    type V: Copy + Eq + Ord + Neg<Output=Self::V>;

    /// Return a value V that such that for any possible value `v`: `-bound <= v <= bound`.
    fn bound(&self) -> Self::V;

    /// Return the heuristic value for the given board from the the next player POV.
    /// `length` is the number of moves played up to this point. Typically used to prefer shorter wins and longer losses.
    /// This value must induce a zero-sum game.
    fn value(&self, board: &B, length: u32) -> Self::V;

    /// Return the value of `child`, given the previous board, its value and the move that was just played.
    /// This function can be overridden to improve performance.
    ///
    /// Given:
    /// * `child = board.clone_and_play(mv)`
    /// * `board_value = value(board, board_length)`
    /// * `child_length = board_length + 1`
    ///
    /// This function must ensure that
    /// * `value(child, child_length) == value_update(board, board_value, board_length, mv, child)`
    #[allow(unused_variables)]
    fn value_update(
        &self,
        board: &B,
        board_value: Self::V,
        board_length: u32,
        mv: B::Move,
        child: &B,
    ) -> Self::V {
        self.value(child, board_length + 1)
    }
}

#[derive(Debug)]
pub struct MinimaxResult<V, M> {
    /// The value of this board.
    pub value: V,

    /// The best move to play, `None` is the board is done or the search depth was 0.
    pub best_move: Option<M>,
}

/// Evaluate the board using minimax with the given heuristic up to the given depth.
/// Return both the value and the best move. If multiple moves have the same value pick a random one using `rng`.
pub fn minimax<B: Board, H: Heuristic<B>>(
    board: &B,
    heuristic: &H,
    depth: u32,
    rng: &mut impl Rng,
) -> MinimaxResult<H::V, B::Move> {
    let result = negamax_recurse(
        heuristic,
        board,
        heuristic.value(board, 0),
        0,
        depth,
        -heuristic.bound(),
        heuristic.bound(),
        RandomBestMoveSelector::new(rng),
    );

    if result.best_move.is_none() {
        assert!(board.is_done() || depth == 0, "Implementation error in negamax");
    }

    result
}

/// Evaluate the board using minimax with the given heuristic up to the given depth.
/// Only returns the value without selecting a move, and so doesn't require an `Rng`.
pub fn minimax_value<B: Board, H: Heuristic<B>>(board: &B, heuristic: &H, depth: u32) -> H::V {
    negamax_recurse(
        heuristic,
        board,
        heuristic.value(board, 0),
        0,
        depth,
        -heuristic.bound(),
        heuristic.bound(),
        NoMoveSelector,
    ).value
}

/// This is a trait so negamax_recurse is instantiated twice,
/// once for the top-level search with move selection and once for deeper nodes without any moves.
trait MoveSelector {
    fn accept(&mut self) -> bool;
}

/// Don't accept any move.
struct NoMoveSelector;

impl MoveSelector for NoMoveSelector {
    fn accept(&mut self) -> bool {
        false
    }
}

/// Implement each move with equal probability,
/// implemented using [reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling).
struct RandomBestMoveSelector<'a, R: Rng> {
    rng: &'a mut R,
    count: u32,
}

impl<'a, R: Rng> RandomBestMoveSelector<'a, R> {
    pub fn new(rng: &'a mut R) -> Self {
        RandomBestMoveSelector { rng, count: 0 }
    }
}

impl<R: Rng> MoveSelector for RandomBestMoveSelector<'_, R> {
    fn accept(&mut self) -> bool {
        self.count += 1;
        self.rng.gen_range(0..self.count) == 0
    }
}

/// The core minimax implementation.
/// Fail-Soft Alpha-Beta Negamax, implementation based on
/// https://www.chessprogramming.org/Alpha-Beta#cite_note-9
fn negamax_recurse<B: Board, H: Heuristic<B>>(
    heuristic: &H,
    board: &B,
    board_heuristic: H::V,
    length: u32,
    depth_left: u32,
    alpha: H::V,
    beta: H::V,
    mut move_selector: impl MoveSelector,
) -> MinimaxResult<H::V, B::Move> {
    if depth_left == 0 || board.is_done() {
        return MinimaxResult { value: board_heuristic, best_move: None };
    }

    let mut best_value = -heuristic.bound();
    let mut best_move: Option<B::Move> = None;
    let mut alpha = alpha;

    let early = board.available_moves().find_map(|mv: B::Move| {
        let child = board.clone_and_play(mv);
        let child_heuristic = heuristic.value_update(board, board_heuristic, length, mv, &child);

        let child_value = -negamax_recurse(
            heuristic,
            &child,
            child_heuristic,
            length + 1,
            depth_left - 1,
            -beta,
            -alpha,
            NoMoveSelector,
        ).value;

        if child_value >= beta {
            //early return, this stops looping over the available moves
            return Some(MinimaxResult { value: child_value, best_move: Some(mv) });
        }

        if child_value >= best_value || best_move.is_none() {
            best_value = child_value;
            alpha = max(alpha, child_value);

            if move_selector.accept() {
                best_move = Some(mv);
            }
        }

        None
    });

    if let Some(early) = early {
        early
    } else {
        MinimaxResult { value: best_value, best_move }
    }
}

pub struct MiniMaxBot<B: Board, H: Heuristic<B>, R: Rng> {
    depth: u32,
    heuristic: H,
    rng: R,
    ph: PhantomData<B>,
}

impl<B: Board, H: Heuristic<B>, R: Rng> Debug for MiniMaxBot<B, H, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MiniMaxBot {{ depth: {}, heuristic: {:?} }}", self.depth, self.heuristic)
    }
}

impl<B: Board, H: Heuristic<B>, R: Rng> MiniMaxBot<B, H, R> {
    pub fn new(depth: u32, heuristic: H, rng: R) -> Self {
        assert!(depth > 0, "requires depth>0 to find the best move");
        MiniMaxBot { depth, heuristic, rng, ph: PhantomData }
    }
}

impl<B: Board, H: Heuristic<B>, R: Rng> Bot<B> for MiniMaxBot<B, H, R> {
    fn select_move(&mut self, board: &B) -> B::Move {
        assert!(!board.is_done());
        minimax(board, &self.heuristic, self.depth, &mut self.rng).best_move.unwrap()
    }
}
