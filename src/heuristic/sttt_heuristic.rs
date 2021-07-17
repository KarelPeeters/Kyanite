use crate::ai::minimax::Heuristic;
use crate::board::{Board, Outcome};
use crate::games::sttt::{Coord, STTTBoard};

#[derive(Debug)]
pub struct STTTTileHeuristic {
    oo_factors: [i32; 3],
    macro_factor: i32,
}

impl Default for STTTTileHeuristic {
    fn default() -> Self {
        STTTTileHeuristic {
            oo_factors: [1, 3, 4],
            macro_factor: 1000,
        }
    }
}

impl Heuristic<STTTBoard> for STTTTileHeuristic {
    type V = i32;

    fn bound(&self) -> i32 {
        i32::MAX
    }

    fn value(&self, board: &STTTBoard, length: u32) -> i32 {
        // win
        if let Some(Outcome::WonBy(player)) = board.outcome() {
            return player.sign(board.next_player()) as i32 * (self.bound() - length as i32);
        }

        // tile
        let tile_value = Coord::all().map(|c| {
            self.oo_factor(c.om()) * self.oo_factor(c.os()) *
                board.tile(c).map_or(0, |p| p.sign(board.next_player()) as i32)
        }).sum::<i32>();

        // macro
        let macr_value = (0..9).map(|om| {
            self.oo_factor(om) *
                board.macr(om).map_or(0, |p| p.sign(board.next_player()) as i32)
        }).sum::<i32>() * self.macro_factor;

        tile_value + macr_value
    }

    fn value_update(&self, board: &STTTBoard, board_value: i32, board_length: u32, mv: Coord, child: &STTTBoard) -> i32 {
        // win
        if let Some(Outcome::WonBy(player)) = board.outcome() {
            return player.sign(board.next_player()) as i32 * (self.bound() - (board_length + 1) as i32);
        }

        let mut neg_child_value = board_value;

        // tile
        neg_child_value += self.oo_factor(mv.om()) * self.oo_factor(mv.os());

        // macro
        if child.macr(mv.om()).is_some() {
            neg_child_value += self.macro_factor * self.oo_factor(mv.om());
        }

        -neg_child_value
    }
}

impl STTTTileHeuristic {
    fn oo_factor(&self, oo: u8) -> i32 {
        let index = match oo {
            1 | 3 | 5 | 7 => 0,
            0 | 2 | 6 | 8 => 1,
            4 => 2,
            _ => panic!("Invalid oo value {}", oo)
        };
        self.oo_factors[index]
    }
}
