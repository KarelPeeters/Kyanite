use board_game::games::ataxx::{AtaxxBoard, Coord, Move};
use board_game::wdl::WDL;
use crate::zero::ZeroEvaluation;
use board_game::board::BoardAvailableMoves;
use internal_iterator::InternalIterator;
use crate::games::ataxx_output::FROM_DX_DY;
use crate::network::softmax;

pub const INPUT_SIZE: usize = 3 * 7 * 7;
pub const POLICY_SIZE: usize = (16 + 1) * 7 * 7;

pub fn encode_input(boards: &[AtaxxBoard]) -> Vec<f32> {
    let mut input = Vec::new();

    for board in boards {
        let (next_tiles, other_tiles) = board.tiles_pov();
        input.extend(Coord::all().map(|c| next_tiles.has(c) as u8 as f32));
        input.extend(Coord::all().map(|c| other_tiles.has(c) as u8 as f32));
        input.extend(Coord::all().map(|c| board.gaps().has(c) as u8 as f32));
    }

    assert_eq!(boards.len() * INPUT_SIZE, input.len());
    input
}

pub fn decode_output(boards: &[AtaxxBoard], batch_wdl_logit: &[f32], batch_policy_logit: &[f32]) -> Vec<ZeroEvaluation> {
    let batch_size = boards.len();
    assert_eq!(batch_size * 3, batch_wdl_logit.len());
    assert_eq!(batch_size * POLICY_SIZE, batch_policy_logit.len());

    boards.iter().enumerate().map(|(bi, board)| {
        // wdl
        let mut wdl = batch_wdl_logit[3 * bi..3 * (bi + 1)].to_vec();
        softmax(&mut wdl);
        let wdl = WDL {
            win: wdl[0],
            draw: wdl[1],
            loss: wdl[2],
        };

        // policy
        let policy_start = bi * 17 * 7 * 7;
        let mut policy: Vec<f32> = board.available_moves().map(|mv| {
            match mv {
                Move::Pass => 1.0,
                Move::Copy { to } => {
                    let to_index = to.dense_i() as usize;
                    batch_policy_logit[policy_start + to_index]
                }
                Move::Jump { from, to } => {
                    let dx = from.x() as i8 - to.x() as i8;
                    let dy = from.y() as i8 - to.y() as i8;
                    let from_index = FROM_DX_DY.iter().position(|&(fdx, fdy)| {
                        fdx == dx && fdy == dy
                    }).unwrap();
                    let to_index = to.dense_i() as usize;
                    batch_policy_logit[policy_start + (1 + from_index) * 7 * 7 + to_index]
                }
            }
        }).collect();
        softmax(&mut policy);

        ZeroEvaluation { wdl, policy }
    }).collect()
}
