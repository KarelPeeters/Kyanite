use board_game::board::Board;
use board_game::games::sttt::{Coord, STTTBoard};

use crate::mapping::{InputMapper, PolicyMapper};
use crate::mapping::bit_buffer::BitBuffer;

#[derive(Debug, Copy, Clone)]
pub struct STTTStdMapper;

impl InputMapper<STTTBoard> for STTTStdMapper {
    const INPUT_BOARD_SIZE: usize = 9;
    const INPUT_BOOL_PLANES: usize = 3;
    const INPUT_SCALAR_COUNT: usize = 0;

    fn encode(&self, bools: &mut BitBuffer, _: &mut Vec<f32>, board: &STTTBoard) {
        bools.extend(Coord::all().map(|c| board.tile(c) == Some(board.next_player())));
        bools.extend(Coord::all().map(|c| board.tile(c) == Some(board.next_player().other())));
        bools.extend(Coord::all().map(|c| board.is_available_move(c)));
    }
}

impl PolicyMapper<STTTBoard> for STTTStdMapper {
    const POLICY_BOARD_SIZE: usize = 9;
    const POLICY_PLANES: usize = 1;

    fn move_to_index(&self, _: &STTTBoard, mv: Coord) -> Option<usize> {
        Some(mv.o() as usize)
    }

    fn index_to_move(&self, _: &STTTBoard, index: usize) -> Option<Coord> {
        assert!(index < 256);
        Some(Coord::from_o(index as u8))
    }
}