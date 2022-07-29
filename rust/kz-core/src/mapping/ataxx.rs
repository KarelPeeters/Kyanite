use board_game::games::ataxx::{AtaxxBoard, Move, MAX_MOVES_SINCE_LAST_COPY};
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
        let size = self.size as usize;

        let index = match mv {
            Move::Pass => (17 * size * size),
            Move::Copy { to } => (to.y() as usize * size + to.x() as usize),
            Move::Jump { from, to } => {
                let dx = from.x() as i8 - to.x() as i8;
                let dy = from.y() as i8 - to.y() as i8;
                let from_index = FROM_DX_DY
                    .iter()
                    .position(|&(fdx, fdy)| fdx == dx && fdy == dy)
                    .unwrap();
                let to_index = to.y() as usize * size + to.x() as usize;

                (1 + from_index) * (size * size) + to_index
            }
        };

        assert!(index < self.policy_len());

        index
    }

    fn index_to_move(&self, board: &AtaxxBoard, index: usize) -> Option<Move> {
        assert_eq!(self.size, board.size());
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
        todo!()
    }

    fn encoded_move_shape(&self) -> [usize; 3] {
        todo!()
    }

    fn encode_mv(&self, _: &mut Vec<f32>, _: usize) {
        todo!()
    }
}
