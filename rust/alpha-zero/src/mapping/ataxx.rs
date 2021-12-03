use board_game::games::ataxx::{AtaxxBoard, Coord, Move};

use crate::mapping::{InputMapper, PolicyMapper};
use crate::mapping::bit_buffer::BitBuffer;

#[derive(Debug, Copy, Clone)]
pub struct AtaxxStdMapper;

impl InputMapper<AtaxxBoard> for AtaxxStdMapper {
    fn input_bool_shape(&self) -> [usize; 3] {
        [3, 7, 7]
    }

    fn input_scalar_count(&self) -> usize {
        0
    }

    fn encode(&self, bools: &mut BitBuffer, _: &mut Vec<f32>, board: &AtaxxBoard) {
        let (next_tiles, other_tiles) = board.tiles_pov();
        bools.extend(Coord::all().map(|c| next_tiles.has(c)));
        bools.extend(Coord::all().map(|c| other_tiles.has(c)));
        bools.extend(Coord::all().map(|c| board.gaps().has(c)));
    }
}

impl PolicyMapper<AtaxxBoard> for AtaxxStdMapper {
    fn policy_shape(&self) -> &[usize] {
        &[17, 7, 7]
    }

    fn move_to_index(&self, _: &AtaxxBoard, mv: Move) -> Option<usize> {
        let index = match mv {
            Move::Pass => None,
            Move::Copy { to } => Some(to.dense_i() as usize),
            Move::Jump { from, to } => {
                let dx = from.x() as i8 - to.x() as i8;
                let dy = from.y() as i8 - to.y() as i8;
                let from_index = FROM_DX_DY.iter().position(|&(fdx, fdy)| {
                    fdx == dx && fdy == dy
                }).unwrap();
                let to_index = to.dense_i() as usize;

                Some((1 + from_index) * 7 * 7 + to_index)
            }
        };

        if let Some(index) = index {
            assert!(index < self.policy_len())
        }

        index
    }

    fn index_to_move(&self, _: &AtaxxBoard, index: usize) -> Option<Move> {
        assert!(index < self.policy_len());

        let to = Coord::from_dense_i((index % (7 * 7)) as u8);

        if index < 7 * 7 {
            Some(Move::Copy { to })
        } else {
            let from_index = index / (7 * 7) - 1;
            let (dx, dy) = FROM_DX_DY[from_index];
            let fx = to.x() as i32 + dx as i32;
            let fy = to.y() as i32 + dy as i32;

            if (0..7).contains(&fx) && (0..7).contains(&fy) {
                let from = Coord::from_xy(fx as u8, fy as u8);
                Some(Move::Jump { from, to })
            } else {
                None
            }
        }
    }
}

pub const FROM_DX_DY: [(i8, i8); 16] = [
    (-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2),
    (-2, -1), (2, -1),
    (-2, 0), (2, 0),
    (-2, 1), (2, 1),
    (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2),
];