use itertools::Itertools;
use itertools::izip;
use sttt::board::{Board, Coord};

use crate::zero::{Request, Response};

pub mod dummy;

#[cfg(feature = "torch")]
pub mod google_torch;
#[cfg(feature = "onnx")]
pub mod google_onnx;

#[derive(Debug)]
pub struct NetworkEvaluation {
    pub value: f32,

    /// The full policy vector, in **o-order**.
    /// Must be zero for non-available moves and have sum `1.0`.
    pub policy: Vec<f32>,
}


pub trait Network {
    fn evaluate_batch(&mut self, boards: &[Board]) -> Vec<NetworkEvaluation>;

    fn evaluate_batch_requests(&mut self, requests: &[Request]) -> Vec<Response> {
        let boards = requests.iter().map(|r| r.board()).collect_vec();
        let evaluations = self.evaluate_batch(&boards);
        izip!(requests, evaluations)
            .map(|(request, evaluation)| Response { request: request.clone(), evaluation })
            .collect_vec()
    }

    fn evaluate(&mut self, board: &Board) -> NetworkEvaluation {
        let mut result = self.evaluate_batch(&[board.clone()]);
        assert_eq!(result.len(), 1);
        result.pop().unwrap()
    }
}

#[allow(dead_code)]
fn encode_google_input(boards: &[Board]) -> Vec<f32> {
    let capacity = boards.len() * 5 * 9 * 9;
    let mut result = Vec::with_capacity(capacity);

    for board in boards {
        result.extend(Coord::all_yx().map(|c| board.is_available_move(c) as u8 as f32));
        result.extend(Coord::all_yx().map(|c| (board.tile(c) == board.next_player) as u8 as f32));
        result.extend(Coord::all_yx().map(|c| (board.tile(c) == board.next_player.other()) as u8 as f32));
        result.extend(Coord::all_yx().map(|c| (board.macr(c.om()) == board.next_player) as u8 as f32));
        result.extend(Coord::all_yx().map(|c| (board.macr(c.om()) == board.next_player.other()) as u8 as f32));
    }

    assert_eq!(capacity, result.len());
    result
}

#[allow(dead_code)]
fn collect_google_output(boards: &[Board], batch_values: &[f32], batch_policies: &[f32]) -> Vec<NetworkEvaluation> {
    assert_eq!(boards.len(), batch_values.len());
    assert_eq!(boards.len() * 81, batch_policies.len());

    boards.iter().enumerate().map(|(i, board)| {
        let range = (81 * i)..(81 * (i + 1));
        let policy_yx = &batch_policies[range];
        let mut policy = Coord::all()
            .map(|c| policy_yx[c.yx() as usize])
            .collect_vec();

        mask_and_softmax(&mut policy, board);

        NetworkEvaluation {
            value: batch_values[i],
            policy,
        }
    }).collect()
}

#[allow(dead_code)]
pub fn mask_and_softmax(slice: &mut [f32], board: &Board) {
    assert_eq!(81, slice.len());

    let mut sum = 0.0;
    for (o, v) in slice.iter_mut().enumerate() {
        if !board.is_available_move(Coord::from_o(o as u8)) {
            *v = 0.0;
        } else {
            *v = v.exp();
            sum += *v;
        }
    }

    for v in slice.iter_mut() {
        *v /= sum;
    }
}