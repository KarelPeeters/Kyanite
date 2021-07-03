use itertools::zip;
use sttt::board::Board;

use crate::network::{mask_and_softmax, Network, NetworkEvaluation, WDL};

/// A `Network` that always returns value and a uniform policy.
pub struct DummyNetwork;

impl Network for DummyNetwork {
    fn evaluate_batch(&mut self, boards: &[Board]) -> Vec<NetworkEvaluation> {
        boards.iter()
            .map(|board| NetworkEvaluation {
                wdl: dummy_wdl(),
                policy: uniform_policy(board),
            })
            .collect()
    }
}

/// A `Network` that returns value 0 and a policy as returned by the inner network.
pub struct DummyValueNetwork<N: Network>(pub N);

impl<N: Network> Network for DummyValueNetwork<N> {
    fn evaluate_batch(&mut self, boards: &[Board]) -> Vec<NetworkEvaluation> {
        let mut result = self.0.evaluate_batch(boards);
        for eval in &mut result {
            eval.wdl = dummy_wdl();
        }
        result
    }
}

/// A `Network` that returns the value returned by the inner network and a uniform policy.
pub struct DummyPolicyNetwork<N: Network>(pub N);

impl<N: Network> Network for DummyPolicyNetwork<N> {
    fn evaluate_batch(&mut self, boards: &[Board]) -> Vec<NetworkEvaluation> {
        let mut result = self.0.evaluate_batch(boards);
        for (board, eval) in zip(boards, &mut result) {
            eval.policy = uniform_policy(board);
        }
        result
    }
}

fn dummy_wdl() -> WDL {
    WDL { win: 1.0 / 3.0, draw: 1.0 / 3.0, loss: 1.0 / 3.0 }
}

fn uniform_policy(board: &Board) -> Vec<f32> {
    let mut policy = vec![0.0; 81];
    mask_and_softmax(&mut policy, board);
    policy
}