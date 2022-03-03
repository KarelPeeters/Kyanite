use std::borrow::{Borrow, Cow};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;

use board_game::board::Board;
use board_game::symmetry::{Symmetry, SymmetryDistribution};
use internal_iterator::InternalIterator;
use itertools::izip;
use rand::distributions::Distribution;
use rand::Rng;

use kz_util::IndexOf;

use crate::network::{Network, ZeroEvaluation};

/// Wrapper around a `Network` that optionally first applies a random symmetry to evaluated boards.
pub struct RandomSymmetryNetwork<B: Board, N: Network<B>, R: Rng> {
    inner: N,
    rng: R,
    enabled: bool,
    ph: PhantomData<B>,
}

impl<B: Board, N: Network<B>, R: Rng> RandomSymmetryNetwork<B, N, R> {
    pub fn new(inner: N, rng: R, enabled: bool) -> Self {
        RandomSymmetryNetwork {
            inner,
            rng,
            enabled,
            ph: PhantomData,
        }
    }
}

impl<B: Board, N: Network<B>, R: Rng> Network<B> for RandomSymmetryNetwork<B, N, R> {
    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation<'static>> {
        if !self.enabled || B::Symmetry::is_unit() {
            // shortcut, just forward to inner if there's nothing to do
            return self.inner.evaluate_batch(boards);
        }

        // map the boards
        let (symmetries, mapped_boards): (Vec<B::Symmetry>, Vec<B>) = boards
            .iter()
            .map(|board| {
                let board = board.borrow();
                let sym = SymmetryDistribution.sample(&mut self.rng);

                (sym, board.map(sym))
            })
            .unzip();

        // call the inner network
        let mapped_evals = self.inner.evaluate_batch(&mapped_boards);

        // un-map the evaluations
        let evals = izip!(boards, symmetries, mapped_boards, mapped_evals)
            .map(|(board, sym, mapped_board, mapped_eval)| unmap_eval(board.borrow(), sym, mapped_board, mapped_eval))
            .collect();

        evals
    }
}

fn unmap_eval<B: Board>(
    board: &B,
    sym: B::Symmetry,
    mapped_board: B,
    mapped_eval: ZeroEvaluation,
) -> ZeroEvaluation<'static> {
    let mapped_moves: Vec<B::Move> = mapped_board.available_moves().collect();

    let policy = board
        .available_moves()
        .map(|mv| {
            let mapped_mv = board.map_move(sym, mv);
            let index = mapped_moves.iter().index_of(&mapped_mv).unwrap();
            mapped_eval.policy[index]
        })
        .collect();

    ZeroEvaluation {
        values: mapped_eval.values,
        policy: Cow::Owned(policy),
    }
}

impl<B: Board, N: Network<B>, R: Rng> Debug for RandomSymmetryNetwork<B, N, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomSymmetryNetwork")
            .field("enabled", &self.enabled)
            .field("inner", &self.inner)
            .finish()
    }
}
