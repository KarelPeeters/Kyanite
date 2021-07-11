use crate::ai::minimax::Heuristic;
use crate::board::{Board, Outcome};
use crate::games::ataxx::{AtaxxBoard, Tiles};

pub struct AtaxxTileHeuristic {
    tile_factor: i32,
    surface_factor: i32,
}

impl AtaxxTileHeuristic {
    pub fn new(tile_factor: i32, surface_factor: i32) -> Self {
        AtaxxTileHeuristic { tile_factor, surface_factor }
    }
}

impl Default for AtaxxTileHeuristic {
    fn default() -> Self {
        AtaxxTileHeuristic {
            tile_factor: 1,
            surface_factor: 0,
        }
    }
}

impl AtaxxTileHeuristic {
    fn player_score(&self, board: &AtaxxBoard, tiles: Tiles) -> i32 {
        let tile_count = tiles.count() as i32;
        let surface_area = (tiles.copy_targets() & board.free_tiles()).count() as i32;

        self.tile_factor * tile_count + self.surface_factor * surface_area
    }
}

impl Heuristic<AtaxxBoard> for AtaxxTileHeuristic {
    type V = i32;

    fn bound(&self) -> Self::V {
        i32::MAX
    }

    fn value(&self, board: &AtaxxBoard) -> Self::V {
        if let Some(outcome) = board.outcome() {
            match outcome {
                Outcome::WonBy(player) => player.sign(board.next_player()) as i32 * i32::MAX,
                Outcome::Draw => 0,
            }
        } else {
            let (next, other) = board.tiles_pov();
            self.player_score(board, next) - self.player_score(board, other)
        }
    }
}
