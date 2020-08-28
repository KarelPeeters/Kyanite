use crate::board::Board;

pub trait Heuristic {
    fn evaluate(&self, board: &Board) -> f32;
}

pub struct ZeroHeuristic;

impl Heuristic for ZeroHeuristic {
    fn evaluate(&self, _: &Board) -> f32 {
        0.0
    }
}

///Gives points for won macros
pub struct MacroHeuristic {
    pub weight: f32,
}

impl Heuristic for MacroHeuristic {
    fn evaluate(&self, board: &Board) -> f32 {
        let mut result = 0;
        for om in 0..9 {
            let p = board.macr(om);

            if p == board.next_player {
                result += 1
            } else if p == board.next_player.other() {
                result -= 1
            }
        }

        //convert -9..9 to 0..self.weight
        ((result as f32) + 9.0) / 18.0 * self.weight
    }
}

///Gives points for how many macros are still needed to get three in a row
pub struct MacroLeftHeuristic {
    pub weight: f32,
}

const LINE_MASKS: [u16; 8] = [
    0b000_000_111,
    0b000_111_000,
    0b111_000_000,
    0b001_001_001,
    0b010_010_010,
    0b100_100_100,
    0b100_010_001,
    0b001_010_100,
];

fn get_min_distance(mask: u16) -> u32 {
    LINE_MASKS.iter().map(|&line_mask| {
        3 - (line_mask & mask).count_ones()
    }).min().unwrap()
}

impl Heuristic for MacroLeftHeuristic {
    fn evaluate(&self, board: &Board) -> f32 {
        let mut next_mask: u16 = 0;
        let mut other_mask: u16 = 0;

        for om in 0..9 {
            let p = board.macr(om);
            if p == board.next_player {
                next_mask |= 1 << om;
            } else if p == board.next_player.other() {
                other_mask |= 1 << om;
            }
        }

        let result = (get_min_distance(other_mask) as i32) - (get_min_distance(next_mask) as i32);

        //map -3..3 to 0..self.weight
        ((result as f32) + 3.0) / 6.0 * self.weight
    }
}