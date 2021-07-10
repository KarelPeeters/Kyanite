use crate::ai::minimax::Heuristic;
use crate::board::Board;
use crate::games::sttt::{Coord, STTTBoard};

pub struct STTTTileHeuristic {
    oo_factors: [f32; 3],
    macro_factor: f32,
}

impl Default for STTTTileHeuristic {
    fn default() -> Self {
        STTTTileHeuristic {
            oo_factors: [1.0, 3.0, 4.0],
            macro_factor: 1000.0,
        }
    }
}

impl Heuristic<STTTBoard> for STTTTileHeuristic {
    fn value(&self, board: &STTTBoard) -> f32 {
        let tile_value = Coord::all().map(|c| {
            self.oo_factor(c.om()) * self.oo_factor(c.os()) *
                board.tile(c).map_or(0.0, |p| p.sign(board.next_player()))
        }).sum::<f32>();

        let macr_value = (0..9).map(|om| {
            self.oo_factor(om) *
                board.macr(om).map_or(0.0, |p| p.sign(board.next_player()))
        }).sum::<f32>() * self.macro_factor;

        let win_value: f32 = board.outcome()
            .map_or(0.0, |p| p.inf_sign(board.next_player()));

        tile_value + macr_value + win_value
    }

    fn value_update(&self, board: &STTTBoard, board_value: f32, mv: Coord, child: &STTTBoard) -> f32 {
        let mut neg_child_value = board_value;

        // tile
        neg_child_value += self.oo_factor(mv.om()) * self.oo_factor(mv.os());

        // macro
        if child.macr(mv.om()).is_some() {
            neg_child_value += self.macro_factor * self.oo_factor(mv.om());
        }

        // win
        if let Some(player) = child.outcome() {
            neg_child_value += player.inf_sign(board.next_player());
        }

        -neg_child_value
    }
}

impl STTTTileHeuristic {
    fn oo_factor(&self, oo: u8) -> f32 {
        let index = match oo {
            1 | 3 | 5 | 7 => 0,
            0 | 2 | 6 | 8 => 1,
            4 => 2,
            _ => panic!("Invalid oo value {}", oo)
        };
        self.oo_factors[index]
    }
}
