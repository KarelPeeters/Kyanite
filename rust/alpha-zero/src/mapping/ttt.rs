use board_game::board::Board;
use board_game::games::ttt::{Coord, TTTBoard};

use crate::mapping::{InputMapper, PolicyMapper};
use crate::mapping::bit_buffer::BitBuffer;

#[derive(Debug, Copy, Clone)]
pub struct TTTStdMapper;

impl InputMapper<TTTBoard> for TTTStdMapper {
    const INPUT_BOARD_SIZE: usize = 3;
    const INPUT_BOOL_PLANES: usize = 2;
    const INPUT_SCALAR_COUNT: usize = 0;

    fn encode(&self, bools: &mut BitBuffer, _: &mut Vec<f32>, board: &TTTBoard) {
        bools.extend(Coord::all().map(|c| board.tile(c) == Some(board.next_player())));
        bools.extend(Coord::all().map(|c| board.tile(c) == Some(board.next_player().other())));
    }
}

impl PolicyMapper<TTTBoard> for TTTStdMapper {
    const POLICY_BOARD_SIZE: usize = 3;
    const POLICY_PLANES: usize = 1;

    fn move_to_index(&self, _: &TTTBoard, mv: Coord) -> Option<usize> {
        Some(mv.i())
    }

    fn index_to_move(&self, _: &TTTBoard, index: usize) -> Option<Coord> {
        Some(Coord::from_i(index))
    }
}