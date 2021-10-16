use std::fmt::Debug;
use std::marker::PhantomData;
use std::panic::{RefUnwindSafe, UnwindSafe};

use board_game::board::Board;

use crate::mapping::bit_buffer::BitBuffer;

pub mod ttt;
pub mod sttt;
pub mod ataxx;
pub mod chess;
pub mod bit_buffer;

pub mod binary_output;
pub mod pgn_to_bin;

/// A way to encode a board as a tensor.
pub trait InputMapper<B: Board>: Debug + Copy + Send + Sync + UnwindSafe + RefUnwindSafe {
    const INPUT_BOARD_SIZE: usize;

    const INPUT_BOOL_PLANES: usize;
    const INPUT_SCALAR_COUNT: usize;

    const INPUT_BOOL_COUNT: usize = Self::INPUT_BOOL_PLANES * Self::INPUT_BOARD_SIZE * Self::INPUT_BOARD_SIZE;

    const INPUT_FULL_PLANES: usize = Self::INPUT_SCALAR_COUNT + Self::INPUT_BOOL_PLANES;
    const INPUT_FULL_SHAPE: [usize; 3] = [Self::INPUT_FULL_PLANES, Self::INPUT_BOARD_SIZE, Self::INPUT_BOARD_SIZE];
    const INPUT_FULL_SIZE: usize = (Self::INPUT_SCALAR_COUNT + Self::INPUT_BOOL_PLANES) * Self::INPUT_BOARD_SIZE * Self::INPUT_BOARD_SIZE;

    /// Encode this board.
    /// Should append `BOOL_COUNT` booleans to `bool_result` and `FLOAT_COUNT` floats to `float_result`..
    fn encode(&self, bools: &mut BitBuffer, scalars: &mut Vec<f32>, board: &B);

    fn encode_full(&self, result: &mut Vec<f32>, board: &B) {
        let mut bools = BitBuffer::new(Self::INPUT_BOOL_COUNT);
        let mut scalars = vec![];

        self.encode(&mut bools, &mut scalars, board);

        assert_eq!(Self::INPUT_BOOL_COUNT, bools.len());
        assert_eq!(Self::INPUT_SCALAR_COUNT, scalars.len());

        let result_start = result.len();

        for s in scalars {
            result.extend(std::iter::repeat(s).take(Self::INPUT_BOARD_SIZE * Self::INPUT_BOARD_SIZE));
        }
        for i in 0..Self::INPUT_BOOL_COUNT {
            result.push(bools[i] as u8 as f32)
        }

        let result_delta = result.len() - result_start;
        assert_eq!(Self::INPUT_FULL_SIZE, result_delta);
    }
}

/// A way to encode and decode moves on a board into a tensor.
pub trait PolicyMapper<B: Board>: Debug + Copy + Send + Sync + UnwindSafe + RefUnwindSafe {
    const POLICY_BOARD_SIZE: usize;
    const POLICY_PLANES: usize;

    const POLICY_SHAPE: [usize; 3] = [Self::POLICY_PLANES, Self::POLICY_BOARD_SIZE, Self::POLICY_BOARD_SIZE];
    const POLICY_SIZE: usize = Self::POLICY_PLANES * Self::POLICY_BOARD_SIZE * Self::POLICY_BOARD_SIZE;

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
    const INPUT_BOARD_SIZE: usize = I::INPUT_BOARD_SIZE;
    const INPUT_BOOL_PLANES: usize = I::INPUT_BOOL_PLANES;
    const INPUT_SCALAR_COUNT: usize = I::INPUT_SCALAR_COUNT;

    fn encode(&self, bools: &mut BitBuffer, scalars: &mut Vec<f32>, board: &B) {
        self.input_mapper.encode(bools, scalars, board)
    }
}

impl<B: Board, I: InputMapper<B>, P: PolicyMapper<B>> PolicyMapper<B> for ComposedMapper<B, I, P> {
    const POLICY_BOARD_SIZE: usize = P::POLICY_BOARD_SIZE;
    const POLICY_PLANES: usize = P::POLICY_PLANES;

    fn move_to_index(&self, board: &B, mv: B::Move) -> Option<usize> {
        self.policy_mapper.move_to_index(board, mv)
    }

    fn index_to_move(&self, board: &B, index: usize) -> Option<B::Move> {
        self.policy_mapper.index_to_move(board, index)
    }
}
