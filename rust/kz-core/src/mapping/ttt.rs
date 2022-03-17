use board_game::board::Board;
use board_game::games::ttt::{Coord, TTTBoard};

use crate::mapping::bit_buffer::BitBuffer;
use crate::mapping::{InputMapper, MuZeroMapper, PolicyMapper};

#[derive(Debug, Copy, Clone)]
pub struct TTTStdMapper;

impl InputMapper<TTTBoard> for TTTStdMapper {
    fn input_bool_shape(&self) -> [usize; 3] {
        [2, 9, 9]
    }

    fn input_scalar_count(&self) -> usize {
        0
    }

    fn encode_input(&self, bools: &mut BitBuffer, _: &mut Vec<f32>, board: &TTTBoard) {
        bools.extend(Coord::all().map(|c| board.tile(c) == Some(board.next_player())));
        bools.extend(Coord::all().map(|c| board.tile(c) == Some(board.next_player().other())));
    }
}

impl PolicyMapper<TTTBoard> for TTTStdMapper {
    fn policy_shape(&self) -> &[usize] {
        &[1, 3, 3]
    }

    fn move_to_index(&self, _: &TTTBoard, mv: Coord) -> Option<usize> {
        Some(mv.i())
    }

    fn index_to_move(&self, _: &TTTBoard, index: usize) -> Option<Coord> {
        Some(Coord::from_i(index))
    }
}

impl MuZeroMapper<TTTBoard> for TTTStdMapper {
    fn encoded_move_shape(&self) -> [usize; 3] {
        todo!()
    }

    fn encode_mv(&self, _: &mut Vec<f32>, _: usize) {
        todo!()
    }
}
