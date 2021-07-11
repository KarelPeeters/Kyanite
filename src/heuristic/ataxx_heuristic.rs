use crate::ai::minimax::Heuristic;
use crate::board::{Board, Outcome};
use crate::games::ataxx::AtaxxBoard;

pub struct AtaxxTileHeuristic;

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
            (next.count() as i32) - (other.count() as i32)
        }
    }
}