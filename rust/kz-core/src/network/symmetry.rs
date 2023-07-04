use std::borrow::{Borrow, Cow};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;

use board_game::board::Board;
use board_game::symmetry::{Symmetry, SymmetryDistribution};
use internal_iterator::InternalIterator;
use itertools::{izip, Itertools};
use rand::distributions::Distribution;
use rand::Rng;

use kz_util::sequence::IndexOf;

use crate::network::{Network, ZeroEvaluation};
use crate::zero::values::ZeroValuesPov;

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
    fn max_batch_size(&self) -> usize {
        self.inner.max_batch_size()
    }

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

/// Wrapper around a `Network` that averages both values and policy over all possible symmetries.
#[derive(Debug)]
pub struct AverageSymmetryNetwork<B: Board, N: Network<B>> {
    inner: N,
    ph: PhantomData<B>,
}

impl<B: Board, N: Network<B>> AverageSymmetryNetwork<B, N> {
    pub fn new(inner: N) -> Self {
        AverageSymmetryNetwork { inner, ph: PhantomData }
    }
}

impl<B: Board, N: Network<B>> Network<B> for AverageSymmetryNetwork<B, N> {
    fn max_batch_size(&self) -> usize {
        // we automatically re-batch, so we don't care about the inner batch size
        // TODO should we return inner.max or inner.max/sym.len() instead?
        usize::MAX
    }

    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation<'static>> {
        if B::Symmetry::is_unit() {
            // shortcut, just forward to inner if there's nothing to do
            return self.inner.evaluate_batch(boards);
        }
        let syms = B::Symmetry::all();

        // map the boards
        let mapped_boards = boards
            .iter()
            .flat_map(|b| syms.iter().map(|&sym| b.borrow().map(sym)))
            .collect_vec();

        // call the inner network
        let mut mapped_evals = Vec::with_capacity(mapped_boards.len());
        mapped_evals.extend(
            mapped_boards
                .chunks(self.inner.max_batch_size())
                .flat_map(|chunk| self.inner.evaluate_batch(chunk)),
        );

        // take the average of the evaluations
        let evals = izip!(
            boards.iter().map(|b| b.borrow()),
            mapped_boards.chunks_exact(syms.len()),
            mapped_evals.chunks_exact(syms.len())
        )
        .map(|(board, mapped_boards, mapped_evals): (&B, &[B], &[ZeroEvaluation])| {
            average_evals(board, syms, mapped_boards, mapped_evals)
        })
        .collect_vec();

        evals
    }
}

fn unmap_eval<B: Board>(
    board: &B,
    sym: B::Symmetry,
    mapped_board: B,
    mapped_eval: ZeroEvaluation,
) -> ZeroEvaluation<'static> {
    let mapped_moves: Vec<B::Move> = mapped_board.available_moves().map_or(vec![], |moves| moves.collect());

    let policy = board.available_moves().map_or(vec![], |moves| {
        moves
            .map(|mv| {
                let mapped_mv = board.map_move(sym, mv);
                let mapped_index = mapped_moves.iter().index_of(&mapped_mv).unwrap();
                mapped_eval.policy[mapped_index]
            })
            .collect()
    });

    ZeroEvaluation {
        values: mapped_eval.values,
        policy: Cow::Owned(policy),
    }
}

fn average_evals<B: Board>(
    board: &B,
    syms: &[B::Symmetry],
    mapped_boards: &[B],
    mapped_evals: &[ZeroEvaluation],
) -> ZeroEvaluation<'static> {
    let values = mapped_evals
        .iter()
        .map(|eval| eval.values)
        .fold(ZeroValuesPov::default(), |a, b| a + b)
        / syms.len() as f32;

    let policy_len = mapped_evals.first().map_or(0, |eval| eval.policy.len());
    let mut policy = vec![0.0; policy_len];

    if policy_len > 0 {
        // TODO repeatedly call available moves instead?
        let board_moves: Vec<B::Move> = board.available_moves().unwrap().collect();

        for (&sym, mapped_board, mapped_eval) in izip!(syms, mapped_boards, mapped_evals) {
            let mapped_moves: Vec<B::Move> = mapped_board.available_moves().unwrap().collect();

            for (i, &mv) in board_moves.iter().enumerate() {
                let mapped_mv = board.map_move(sym, mv);
                let mapped_index = mapped_moves.iter().index_of(&mapped_mv).unwrap();
                policy[i] += mapped_eval.policy[mapped_index] / syms.len() as f32;
            }
        }
    }

    ZeroEvaluation {
        values,
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
