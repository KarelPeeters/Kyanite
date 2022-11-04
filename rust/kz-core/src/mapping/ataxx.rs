use board_game::games::ataxx::{AtaxxBoard, Move, MAX_MOVES_SINCE_LAST_COPY};
use board_game::util::bitboard::BitBoard8;
use board_game::util::coord::Coord8;

use crate::mapping::bit_buffer::BitBuffer;
use crate::mapping::{InputMapper, MuZeroMapper, PolicyMapper};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct AtaxxStdMapper {
    size: u8,
    policy_shape: [usize; 1],
}

impl AtaxxStdMapper {
    pub fn new(size: u8) -> Self {
        let policy_size = (17 * size as usize * size as usize) + 1;
        AtaxxStdMapper {
            size,
            policy_shape: [policy_size],
        }
    }

    pub fn size(&self) -> u8 {
        self.size
    }

    pub fn index_to_move(&self, index: usize) -> Option<Move> {
        assert!(index < self.policy_len());

        let size = self.size as usize;
        let area = size * size;
        let to_index = (index % area) as u8;
        let to = Coord8::from_xy(to_index % size as u8, to_index / size as u8);

        if index == self.policy_len() - 1 {
            Some(Move::Pass)
        } else if index < area {
            Some(Move::Copy { to })
        } else {
            let from_index = index / area - 1;
            let (dx, dy) = FROM_DX_DY[from_index];
            let fx = to.x() as i32 + dx as i32;
            let fy = to.y() as i32 + dy as i32;

            if (0..size as i32).contains(&fx) && (0..size as i32).contains(&fy) {
                let from = Coord8::from_xy(fx as u8, fy as u8);
                Some(Move::Jump { from, to })
            } else {
                None
            }
        }
    }

    pub fn move_to_index(&self, mv: Move) -> usize {
        let size = self.size as usize;

        let index = match mv {
            Move::Pass => 17 * size * size,
            Move::Copy { to } => to.dense_index(size as u8),
            Move::Jump { from, to } => {
                let dx = from.x() as i8 - to.x() as i8;
                let dy = from.y() as i8 - to.y() as i8;
                let from_index = FROM_DX_DY
                    .iter()
                    .position(|&(fdx, fdy)| fdx == dx && fdy == dy)
                    .unwrap();
                let to_index = to.dense_index(size as u8);

                (1 + from_index) * (size * size) + to_index
            }
        };

        assert!(index < self.policy_len());
        index
    }

    pub fn encode_mv_split(mv: Option<Move>) -> (bool, Option<Coord8>, Option<Coord8>, Option<Coord8>) {
        match mv {
            None => (false, None, None, None),
            Some(Move::Pass) => (true, None, None, None),
            Some(Move::Copy { to }) => (false, Some(to), None, None),
            Some(Move::Jump { from, to }) => (false, None, Some(from), Some(to)),
        }
    }
}

impl InputMapper<AtaxxBoard> for AtaxxStdMapper {
    fn input_bool_shape(&self) -> [usize; 3] {
        [3, self.size as usize, self.size as usize]
    }

    fn input_scalar_count(&self) -> usize {
        1
    }

    fn encode_input(&self, bools: &mut BitBuffer, scalars: &mut Vec<f32>, board: &AtaxxBoard) {
        scalars.push(board.moves_since_last_copy() as f32 / MAX_MOVES_SINCE_LAST_COPY as f32);

        let (next_tiles, other_tiles) = board.tiles_pov();
        bools.extend(board.full_mask().into_iter().map(|c| next_tiles.has(c)));
        bools.extend(board.full_mask().into_iter().map(|c| other_tiles.has(c)));
        bools.extend(board.full_mask().into_iter().map(|c| board.gaps().has(c)));
    }
}

impl PolicyMapper<AtaxxBoard> for AtaxxStdMapper {
    fn policy_shape(&self) -> &[usize] {
        &self.policy_shape
    }

    fn move_to_index(&self, board: &AtaxxBoard, mv: Move) -> usize {
        assert_eq!(board.size(), self.size);
        self.move_to_index(mv)
    }

    fn index_to_move(&self, board: &AtaxxBoard, index: usize) -> Option<Move> {
        assert_eq!(self.size, board.size());
        self.index_to_move(index)
    }
}

pub const FROM_DX_DY: [(i8, i8); 16] = [
    (-2, -2),
    (-1, -2),
    (0, -2),
    (1, -2),
    (2, -2),
    (-2, -1),
    (2, -1),
    (-2, 0),
    (2, 0),
    (-2, 1),
    (2, 1),
    (-2, 2),
    (-1, 2),
    (0, 2),
    (1, 2),
    (2, 2),
];

impl MuZeroMapper<AtaxxBoard> for AtaxxStdMapper {
    fn state_board_size(&self) -> usize {
        self.size as usize
    }

    fn encoded_move_shape(&self) -> [usize; 3] {
        let size = self.size as usize;
        // planes: (pass, copy to, jump from, jump to)
        [4, size, size]
    }

    fn encode_mv(&self, result: &mut Vec<f32>, mv_index: usize) {
        let mv = self.index_to_move(mv_index);
        let size = self.size;

        let (pass, copy_to, jump_from, jump_to) = Self::encode_mv_split(mv);

        let full = BitBoard8::FULL_FOR_SIZE[size as usize];
        let pass_board = if pass { full } else { BitBoard8::EMPTY };

        append_bitboard(result, size, pass_board);
        append_bitboard(result, size, BitBoard8::coord_option(copy_to));
        append_bitboard(result, size, BitBoard8::coord_option(jump_from));
        append_bitboard(result, size, BitBoard8::coord_option(jump_to));
    }
}

fn append_bitboard(result: &mut Vec<f32>, size: u8, board: BitBoard8) {
    for coord in BitBoard8::FULL_FOR_SIZE[size as usize] {
        result.push(board.has(coord) as u8 as f32);
    }
}
