use board_game::board::Board;
use board_game::games::ttt::TTTBoard;
use board_game::util::coord::Coord3;

use crate::mapping::bit_buffer::BitBuffer;
use crate::mapping::{InputMapper, MuZeroMapper, PolicyMapper};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct TTTStdMapper;

impl InputMapper<TTTBoard> for TTTStdMapper {
    fn input_bool_shape(&self) -> [usize; 3] {
        [2, 3, 3]
    }

    fn input_scalar_count(&self) -> usize {
        0
    }

    fn encode_input(&self, bools: &mut BitBuffer, _: &mut Vec<f32>, board: &TTTBoard) {
        bools.extend(Coord3::all().map(|c| board.tile(c) == Some(board.next_player())));
        bools.extend(Coord3::all().map(|c| board.tile(c) == Some(board.next_player().other())));
    }
}

impl PolicyMapper<TTTBoard> for TTTStdMapper {
    fn policy_shape(&self) -> &[usize] {
        &[1, 3, 3]
    }

    fn move_to_index(&self, _: &TTTBoard, mv: Coord3) -> Option<usize> {
        Some(mv.index() as usize)
    }

    fn index_to_move(&self, _: &TTTBoard, index: usize) -> Option<Coord3> {
        assert!(index < 9);
        Some(Coord3::from_index(index as u8))
    }
}

impl MuZeroMapper<TTTBoard> for TTTStdMapper {
    fn state_board_size(&self) -> usize {
        3
    }

    fn encoded_move_shape(&self) -> [usize; 3] {
        [1, 3, 3]
    }

    fn encode_mv(&self, result: &mut Vec<f32>, mv_index: usize) {
        assert!(mv_index < 9);
        let mv = Coord3::from_index(mv_index as u8);
        result.extend(Coord3::all().map(|c| (c == mv) as u8 as f32))
    }
}
