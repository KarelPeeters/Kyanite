use board_game::board::Board;
use board_game::games::chess::ChessBoard;
use board_game::wdl::OutcomeWDL;
use shakmaty::{CastlingMode, Chess};
use shakmaty::fen::Fen;
use shakmaty_syzygy::{SyzygyError, Tablebase, Wdl};

use crate::oracle::{Oracle, OracleEvaluation};

#[derive(Debug)]
pub struct SyzygyOracle {
    tables: Tablebase<Chess>,
    max_pieces: u32,
}

impl SyzygyOracle {
    pub fn new(tables: Tablebase<Chess>, max_pieces: u32) -> Self {
        SyzygyOracle { tables, max_pieces }
    }
}

impl Oracle<ChessBoard> for SyzygyOracle {
    //TODO override best_outcome here, that should be faster then figuring out the best move as well

    fn evaluate(&self, board: &ChessBoard) -> Option<OracleEvaluation<ChessBoard>> {
        if let Some(outcome) = board.outcome() {
            return Some(OracleEvaluation { best_outcome: outcome, best_move: None });
        }

        //TODO we could check for exact material combination support,
        // but then we'd need a getter for it on TableBase
        if board.inner().combined().popcnt() > self.max_pieces {
            return None;
        }

        //TODO properly implement 50-move rule?
        //  wait for response on https://github.com/niklasf/shakmaty-syzygy/issues/14

        let pos: Chess = board.inner().to_string()
            .parse::<Fen>().unwrap()
            .into_position(CastlingMode::Standard).unwrap();

        match self.tables.best_move(&pos) {
            Ok(Some((best_move, dtz))) => {
                let best_move = board.parse_move(&best_move.to_uci(CastlingMode::Standard).to_string()).unwrap();

                let best_wdl = match Wdl::from_dtz_after_zeroing(dtz) {
                    Wdl::Loss => OutcomeWDL::Loss,
                    Wdl::BlessedLoss | Wdl::Draw | Wdl::CursedWin => OutcomeWDL::Draw,
                    Wdl::Win => OutcomeWDL::Win,
                };
                let best_outcome = best_wdl.un_pov(board.next_player().other());

                Some(OracleEvaluation { best_outcome, best_move: Some(best_move) })
            }
            Ok(None) | Err(SyzygyError::Castling | SyzygyError::TooManyPieces) => None,
            Err(e) => {
                panic!("Error while probing: {:?}", e);
            }
        }
    }
}