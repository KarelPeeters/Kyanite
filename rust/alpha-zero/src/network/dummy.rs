use std::marker::PhantomData;

use internal_iterator::InternalIterator;
use itertools::zip;
use sttt::board::Board;
use sttt::wdl::WDL;

use crate::network::Network;
use crate::zero::ZeroEvaluation;

/// A `Network` that always returns value and a uniform policy.
#[derive(Debug)]
pub struct DummyNetwork;

/// A `Network` that returns value 0 and a policy as returned by the inner network.
#[derive(Debug)]
pub struct DummyValueNetwork<B: Board, N: Network<B>> {
    inner: N,
    ph: PhantomData<*const B>,
}

/// A `Network` that returns the value returned by the inner network and a uniform policy.
#[derive(Debug)]
pub struct DummyPolicyNetwork<B: Board, N: Network<B>> {
    inner: N,
    ph: PhantomData<*const B>,
}

impl<B: Board, N: Network<B>> DummyValueNetwork<B, N> {
    pub fn new(inner: N) -> Self {
        DummyValueNetwork { inner, ph: PhantomData }
    }
}

impl<B: Board, N: Network<B>> DummyPolicyNetwork<B, N> {
    pub fn new(inner: N) -> Self {
        DummyPolicyNetwork { inner, ph: PhantomData }
    }
}

impl<B: Board> Network<B> for DummyNetwork {
    fn evaluate_batch(&mut self, boards: &[B]) -> Vec<ZeroEvaluation> {
        boards.iter()
            .map(|board| ZeroEvaluation {
                wdl: uniform_wdl(),
                policy: uniform_policy(board),
            })
            .collect()
    }
}

impl<B: Board, N: Network<B>> Network<B> for DummyValueNetwork<B, N> {
    fn evaluate_batch(&mut self, boards: &[B]) -> Vec<ZeroEvaluation> {
        let mut result = self.inner.evaluate_batch(boards);
        for eval in &mut result {
            eval.wdl = uniform_wdl();
        }
        result
    }
}

impl<B: Board, N: Network<B>> Network<B> for DummyPolicyNetwork<B, N> {
    fn evaluate_batch(&mut self, boards: &[B]) -> Vec<ZeroEvaluation> {
        let mut result = self.inner.evaluate_batch(boards);
        for (board, eval) in zip(boards, &mut result) {
            eval.policy = uniform_policy(board);
        }
        result
    }
}

fn uniform_wdl() -> WDL<f32> {
    WDL { win: 1.0 / 3.0, draw: 1.0 / 3.0, loss: 1.0 / 3.0 }
}

fn uniform_policy<B: Board>(board: &B) -> Vec<f32> {
    let move_count = board.available_moves().count();
    vec![1.0 / move_count as f32; move_count]
}