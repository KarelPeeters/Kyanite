use itertools::Itertools;
use itertools::izip;
use sttt::board::{Board, Coord};

use crate::zero::{Request, Response};

pub mod dummy;

#[cfg(feature = "torch")]
pub mod google_torch;
#[cfg(feature = "onnx")]
pub mod google_onnx;

#[derive(Default, Debug, Copy, Clone)]
pub struct WDL {
    pub win: f32,
    pub draw: f32,
    pub loss: f32,
}

#[derive(Debug)]
pub struct NetworkEvaluation {
    /// The win, draw and loss probabilities, after normalization.
    pub wdl: WDL,

    /// The full policy probability vector, in **o-order**, after masking and normalization.
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
fn collect_google_output(boards: &[Board], batch_wdl: &[f32], batch_policies: &[f32]) -> Vec<NetworkEvaluation> {
    assert_eq!(boards.len() * 3, batch_wdl.len());
    assert_eq!(boards.len() * 81, batch_policies.len());

    boards.iter().enumerate().map(|(i, board)| {
        let policy_range = (81 * i)..(81 * (i + 1));
        let policy_yx = &batch_policies[policy_range];
        let mut policy = Coord::all()
            .map(|c| policy_yx[c.yx() as usize])
            .collect_vec();

        mask_and_softmax(&mut policy, board);

        let mut wdl = batch_wdl[3 * i..(3 * i + 3)].to_vec();
        softmax(&mut wdl);
        let wdl = WDL { win: wdl[0], draw: wdl[1], loss: wdl[2] };

        NetworkEvaluation { wdl, policy }
    }).collect()
}

#[allow(dead_code)]
pub fn softmax(slice: &mut [f32]) {
    let mut sum = 0.0;
    for v in slice.iter_mut() {
        *v = v.exp();
        sum += *v;
    }
    for v in slice.iter_mut() {
        *v /= sum;
    }
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

impl WDL {
    pub fn nan() -> WDL {
        WDL { win: f32::NAN, draw: f32::NAN, loss: f32::NAN }
    }

    pub fn value(self) -> f32 {
        self.win - self.loss
    }
}

impl std::ops::Neg for WDL {
    type Output = WDL;

    fn neg(self) -> WDL {
        WDL { win: self.loss, draw: self.draw, loss: self.win }
    }
}

impl std::ops::Add<WDL> for WDL {
    type Output = WDL;

    fn add(self, rhs: WDL) -> WDL {
        WDL {
            win: self.win + rhs.win,
            draw: self.draw + rhs.draw,
            loss: self.loss + rhs.loss,
        }
    }
}

impl std::ops::Div<f32> for WDL {
    type Output = WDL;

    fn div(self, rhs: f32) -> WDL {
        WDL {
            win: self.win / rhs,
            draw: self.draw / rhs,
            loss: self.loss / rhs,
        }
    }
}

impl std::ops::AddAssign<WDL> for WDL {
    fn add_assign(&mut self, rhs: WDL) {
        *self = *self + rhs
    }
}