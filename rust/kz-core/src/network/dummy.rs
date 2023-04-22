use std::borrow::{Borrow, Cow};
use std::marker::PhantomData;

use board_game::board::Board;
use board_game::games::max_length::MaxMovesBoard;
use board_game::pov::ScalarPov;
use board_game::wdl::WDL;
use internal_iterator::InternalIterator;
use itertools::{Either, Itertools};

use crate::network::{Network, ZeroEvaluation};
use crate::zero::values::ZeroValuesPov;

/// A `Network` that always returns uniform wdl and policy..
#[derive(Debug, Clone, Copy)]
pub struct DummyNetwork;

/// A `Network` wrapper that returns uniform wdl and the policy as evaluated by the inner network.
#[derive(Debug, Clone, Copy)]
pub struct DummyValueNetwork<B: Board, N: Network<B>> {
    inner: N,
    ph: PhantomData<*const B>,
}

/// A `Network` wrapper that returns wdl evaluated by the inner network and uniform policy.
#[derive(Debug, Clone, Copy)]
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
    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation<'static>> {
        boards
            .iter()
            .map(|board| ZeroEvaluation {
                values: uniform_values(),
                policy: Cow::Owned(uniform_policy(
                    board.borrow().available_moves().map_or(0, |moves| moves.count()),
                )),
            })
            .collect()
    }
}

impl<B: Board, N: Network<B>> Network<B> for DummyValueNetwork<B, N> {
    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation<'static>> {
        self.inner
            .evaluate_batch(boards)
            .into_iter()
            .map(|orig| ZeroEvaluation {
                values: uniform_values(),
                policy: orig.policy,
            })
            .collect()
    }
}

impl<B: Board, N: Network<B>> Network<B> for DummyPolicyNetwork<B, N> {
    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation<'static>> {
        self.inner
            .evaluate_batch(boards)
            .into_iter()
            .map(|orig| ZeroEvaluation {
                values: orig.values,
                policy: Cow::Owned(uniform_policy(orig.policy.len())),
            })
            .collect()
    }
}

pub fn uniform_values() -> ZeroValuesPov {
    ZeroValuesPov {
        value: ScalarPov::new(0.0),
        wdl: WDL {
            win: 1.0 / 3.0,
            draw: 1.0 / 3.0,
            loss: 1.0 / 3.0,
        },
        moves_left: 0.0,
    }
}

pub fn uniform_policy(available_moves: usize) -> Vec<f32> {
    vec![1.0 / available_moves as f32; available_moves]
}

/// A `Network` wrapper that accepts `MaxMovesBoard<B>` instead of `B`, and just passes the inner board along.
///
/// **Warning:** This means the network doesn't get full game state information, which may be undesirable.
#[derive(Debug, Clone, Copy)]
pub struct MaxMovesNetwork<N>(pub N);

impl<N: Network<B>, B: Board> Network<MaxMovesBoard<B>> for MaxMovesNetwork<N> {
    fn evaluate_batch(&mut self, boards: &[impl Borrow<MaxMovesBoard<B>>]) -> Vec<ZeroEvaluation<'static>> {
        // TODO memory allocation, look into changing network so it accepts an iterator
        //   maybe instead of that (since it generates a lot of extra code), accept already-encoded boards as an input?
        let inner_boards = boards.iter().map(|b| b.borrow().inner()).collect_vec();

        self.0.evaluate_batch(&inner_boards)
    }
}

pub type NetworkOrDummy<N> = Either<N, DummyNetwork>;

impl<L: Network<B>, R: Network<B>, B: Board> Network<B> for Either<L, R> {
    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation<'static>> {
        match self {
            Either::Left(left) => left.evaluate_batch(boards),
            Either::Right(right) => right.evaluate_batch(boards),
        }
    }
}
