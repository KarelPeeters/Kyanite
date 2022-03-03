use board_game::games::ataxx::{AtaxxBoard, Coord, Move};

use crate::mapping::{InputMapper, PolicyMapper};
use crate::mapping::bit_buffer::BitBuffer;

#[derive(Debug, Copy, Clone)]
pub struct AtaxxStdMapper {
    size: u8,
    policy_shape: [usize; 3],
}

impl AtaxxStdMapper {
    pub fn new(size: u8) -> Self {
        AtaxxStdMapper { size, policy_shape: [17, size as usize, size as usize] }
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
        0
    }

    fn encode(&self, bools: &mut BitBuffer, _: &mut Vec<f32>, board: &AtaxxBoard) {
        let (next_tiles, other_tiles) = board.tiles_pov();
        bools.extend(Coord::all(self.size).map(|c| next_tiles.has(c)));
        bools.extend(Coord::all(self.size).map(|c| other_tiles.has(c)));
        bools.extend(Coord::all(self.size).map(|c| board.gaps().has(c)));
    }
}

impl PolicyMapper<AtaxxBoard> for AtaxxStdMapper {
    fn policy_shape(&self) -> &[usize] {
        &self.policy_shape
    }

    fn move_to_index(&self, board: &AtaxxBoard, mv: Move) -> Option<usize> {
        let size = board.size();
        assert_eq!(size, self.size);

        let index = match mv {
            Move::Pass => None,
            Move::Copy { to } => Some((to.y() * size + to.x()) as usize),
            Move::Jump { from, to } => {
                let dx = from.x() as i8 - to.x() as i8;
                let dy = from.y() as i8 - to.y() as i8;
                let from_index = FROM_DX_DY.iter().position(|&(fdx, fdy)| {
                    fdx == dx && fdy == dy
                }).unwrap();
                let to_index = (to.y() * size + to.x()) as usize;

                Some((1 + from_index) * (size * size) as usize + to_index)
            }
        };

        if let Some(index) = index {
            assert!(index < self.policy_len())
        }

        index
    }

    fn index_to_move(&self, board: &AtaxxBoard, index: usize) -> Option<Move> {
        let size = board.size();
        assert_eq!(size, self.size);
        assert!(index < self.policy_len());

        let area = (size * size) as usize;
        let to_index = (index % area) as u8;
        let to = Coord::from_xy(to_index % size, to_index / size);

        if index < area {
            Some(Move::Copy { to })
        } else {
            let from_index = index / area - 1;
            let (dx, dy) = FROM_DX_DY[from_index];
            let fx = to.x() as i32 + dx as i32;
            let fy = to.y() as i32 + dy as i32;

            if (0..size as i32).contains(&fx) && (0..size as i32).contains(&fy) {
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