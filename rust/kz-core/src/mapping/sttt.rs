use board_game::board::Board;
use board_game::games::sttt::{Coord, STTTBoard};

use crate::mapping::{InputMapper, PolicyMapper};
use crate::mapping::bit_buffer::BitBuffer;

#[derive(Debug, Copy, Clone)]
pub struct STTTStdMapper;

impl InputMapper<STTTBoard> for STTTStdMapper {
    fn input_bool_shape(&self) -> [usize; 3] {
        [3, 9, 9]
    }

    fn input_scalar_count(&self) -> usize {
        0
    }

    fn encode(&self, bools: &mut BitBuffer, _: &mut Vec<f32>, board: &STTTBoard) {
        bools.extend(Coord::all().map(|c| board.tile(c) == Some(board.next_player())));
        bools.extend(Coord::all().map(|c| board.tile(c) == Some(board.next_player().other())));
        bools.extend(Coord::all().map(|c| board.is_available_move(c)));
    }
}

impl PolicyMapper<STTTBoard> for STTTStdMapper {
    fn policy_shape(&self) -> &[usize] {
        &[1, 9, 9]
    }

    fn move_to_index(&self, _: &STTTBoard, mv: Coord) -> Option<usize> {
        Some(mv.o() as usize)
    }

    fn index_to_move(&self, _: &STTTBoard, index: usize) -> Option<Coord> {
        assert!(index < 256);
        Some(Coord::from_o(index as u8))
    }
}