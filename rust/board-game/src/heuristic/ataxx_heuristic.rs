use crate::ai::minimax::Heuristic;
use crate::ai::solver::SolverHeuristic;
use crate::board::Board;
use crate::games::ataxx::{AtaxxBoard, Tiles};

#[derive(Debug)]
pub struct AtaxxTileHeuristic {
    tile_factor: i32,
    surface_factor: i32,
}

impl AtaxxTileHeuristic {
    pub fn new(tile_factor: i32, surface_factor: i32) -> Self {
        AtaxxTileHeuristic { tile_factor, surface_factor }
    }

    pub fn greedy() -> Self {
        AtaxxTileHeuristic { tile_factor: 1, surface_factor: 0 }
    }
}

impl Default for AtaxxTileHeuristic {
    fn default() -> Self {
        AtaxxTileHeuristic {
            tile_factor: 100,
            surface_factor: 10,
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

    fn value(&self, board: &AtaxxBoard, length: u32) -> Self::V {
        if let Some(_) = board.outcome() {
            // return near-max values for wins/draws/losses
            SolverHeuristic.value(board, length)
        } else {
            let (next, other) = board.tiles_pov();
            self.player_score(board, next) - self.player_score(board, other)
        }
    }
}
