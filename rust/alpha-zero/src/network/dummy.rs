use std::borrow::Borrow;
use std::marker::PhantomData;

use board_game::board::Board;
use board_game::wdl::WDL;
use internal_iterator::InternalIterator;
use itertools::zip;

use crate::network::{Network, ZeroEvaluation};

/// A `Network` that always returns uniform wdl and policy..
#[derive(Debug)]
pub struct DummyNetwork;

/// A `Network` wrapper that returns uniform wdl and the policy as evaluated by the inner network.
#[derive(Debug)]
pub struct DummyValueNetwork<B: Board, N: Network<B>> {
    inner: N,
    ph: PhantomData<*const B>,
}

/// A `Network` wrapper that returns wdl evaluated by the inner network and uniform policy.
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
    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation> {
        boards.iter()
            .map(|board| ZeroEvaluation {
                wdl: uniform_wdl(),
                policy: uniform_policy(board.borrow()),
            })
            .collect()
    }
}

impl<B: Board, N: Network<B>> Network<B> for DummyValueNetwork<B, N> {
    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation> {
        let mut result = self.inner.evaluate_batch(boards);
        for eval in &mut result {
            eval.wdl = uniform_wdl();
        }
        result
    }
}

impl<B: Board, N: Network<B>> Network<B> for DummyPolicyNetwork<B, N> {
    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation> {
        let mut result = self.inner.evaluate_batch(boards);
        for (board, eval) in zip(boards, &mut result) {
            eval.policy = uniform_policy(board.borrow());
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