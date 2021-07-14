use std::fmt::Debug;

use itertools::Itertools;
use itertools::izip;
use sttt::board::Board;

use crate::evaluation::ZeroEvaluation;
use crate::zero::{Request, Response};

pub mod dummy;
#[cfg(feature = "tch")]
pub mod torch_utils;

pub trait Network<B: Board>: Debug {
    fn evaluate_batch(&mut self, boards: &[B]) -> Vec<ZeroEvaluation>;

    fn evaluate_batch_requests(&mut self, requests: &[Request<B>]) -> Vec<Response<B>> {
        let boards = requests.iter().map(|r| r.board()).collect_vec();
        let evaluations = self.evaluate_batch(&boards);
        izip!(requests, evaluations)
            .map(|(request, evaluation)| Response { request: request.clone(), evaluation })
            .collect_vec()
    }

    fn evaluate(&mut self, board: &B) -> ZeroEvaluation {
        let mut result = self.evaluate_batch(&[board.clone()]);
        assert_eq!(result.len(), 1);
        result.pop().unwrap()
    }
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
