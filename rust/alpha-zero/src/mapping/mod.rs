use std::fmt::Debug;
use std::marker::PhantomData;
use std::panic::{RefUnwindSafe, UnwindSafe};

use board_game::board::Board;

pub mod ataxx;
pub mod chess;

pub mod binary_output;

/// A way to encode a board as a tensor.
pub trait InputMapper<B: Board>: Debug + Copy + Send + Sync + UnwindSafe + RefUnwindSafe {
    const INPUT_SHAPE: [usize; 3];
    const INPUT_SIZE: usize = Self::INPUT_SHAPE[0] * Self::INPUT_SHAPE[1] * Self::INPUT_SHAPE[2];

    /// Encode this board, appending the resulting `INPUT_SIZE` values to `output`.
    fn append_board_to(&self, result: &mut Vec<f32>, board: &B);
}

/// A way to encode and decode moves on a board into a tensor.
pub trait PolicyMapper<B: Board>: Debug + Copy + Send + Sync + UnwindSafe + RefUnwindSafe {
    const POLICY_SHAPE: [usize; 3];
    const POLICY_SIZE: usize = Self::POLICY_SHAPE[0] * Self::POLICY_SHAPE[1] * Self::POLICY_SHAPE[2];

    /// Get the index in the policy tensor corresponding to the given move.
    /// A return of `None` means that this move is structurally forced and does not get a place in the policy tensor.
    fn move_to_index(&self, board: &B, mv: B::Move) -> Option<usize>;

    /// Get the move corresponding to the given index in the policy tensor.
    /// A return of `None` means that this index does not correspond to any move.
    fn index_to_move(&self, board: &B, index: usize) -> Option<B::Move>;
}

/// Utility trait automatically implemented for anything that implements both [InputMapper] and [PolicyMapper].
pub trait BoardMapper<B: Board>: InputMapper<B> + PolicyMapper<B> {}

impl<B: Board, M: InputMapper<B> + PolicyMapper<B>> BoardMapper<B> for M {}

/// A [BoardMapper] composed of separate input and policy mappers.
#[derive(Debug)]
pub struct ComposedMapper<B: Board, I: InputMapper<B>, P: PolicyMapper<B>> {
    input_mapper: I,
    policy_mapper: P,
    ph: PhantomData<B>,
}

impl<B: Board, I: InputMapper<B>, P: PolicyMapper<B>> ComposedMapper<B, I, P> {
    pub fn new(input_mapper: I, policy_mapper: P) -> Self {
        ComposedMapper { input_mapper, policy_mapper, ph: PhantomData }
    }
}

impl<B: Board, I: InputMapper<B>, P: PolicyMapper<B>> Clone for ComposedMapper<B, I, P> {
    fn clone(&self) -> Self {
        ComposedMapper {
            input_mapper: self.input_mapper.clone(),
            policy_mapper: self.policy_mapper.clone(),
            ph: PhantomData,
        }
    }
}

impl<B: Board, I: InputMapper<B>, P: PolicyMapper<B>> Copy for ComposedMapper<B, I, P> {}

impl<B: Board, I: InputMapper<B>, P: PolicyMapper<B>> InputMapper<B> for ComposedMapper<B, I, P> {
    const INPUT_SHAPE: [usize; 3] = I::INPUT_SHAPE;

    fn append_board_to(&self, result: &mut Vec<f32>, board: &B) {
        self.input_mapper.append_board_to(result, board)
    }
}

impl<B: Board, I: InputMapper<B>, P: PolicyMapper<B>> PolicyMapper<B> for ComposedMapper<B, I, P> {
    const POLICY_SHAPE: [usize; 3] = P::POLICY_SHAPE;

    fn move_to_index(&self, board: &B, mv: B::Move) -> Option<usize> {
        self.policy_mapper.move_to_index(board, mv)
    }

    fn index_to_move(&self, board: &B, index: usize) -> Option<B::Move> {
        self.policy_mapper.index_to_move(board, index)
    }
}
