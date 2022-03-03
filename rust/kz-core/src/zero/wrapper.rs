use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;

use board_game::ai::Bot;
use board_game::board::Board;
use itertools::Itertools;

use kz_util::zip_eq_exact;

use crate::network::Network;
use crate::oracle::Oracle;
use crate::zero::node::UctWeights;
use crate::zero::step::{zero_step_apply, zero_step_gather, FpuMode};
use crate::zero::tree::Tree;

#[derive(Debug, Copy, Clone)]
pub struct ZeroSettings {
    pub batch_size: usize,
    pub weights: UctWeights,
    pub use_value: bool,
    pub fpu_mode: FpuMode,
}

impl ZeroSettings {
    pub fn new(batch_size: usize, weights: UctWeights, use_value: bool, fpu_mode: FpuMode) -> Self {
        ZeroSettings {
            batch_size,
            weights,
            use_value,
            fpu_mode,
        }
    }
}

impl ZeroSettings {
    /// Construct a new tree from scratch on the given board.
    pub fn build_tree<B: Board>(
        self,
        root_board: &B,
        network: &mut impl Network<B>,
        oracle: &impl Oracle<B>,
        stop: impl FnMut(&Tree<B>) -> bool,
    ) -> Tree<B> {
        let mut tree = Tree::new(root_board.clone());
        self.expand_tree(&mut tree, network, oracle, stop);
        tree
    }

    // Continue expanding an existing tree.
    pub fn expand_tree<B: Board>(
        self,
        tree: &mut Tree<B>,
        network: &mut impl Network<B>,
        oracle: &impl Oracle<B>,
        mut stop: impl FnMut(&Tree<B>) -> bool,
    ) {
        'outer: loop {
            if stop(tree) {
                break 'outer;
            }

            // collect requests until the batch is full or we repeatedly fail to find new positions to evaluate
            let mut requests = vec![];
            let mut terminal_gathers = 0;

            while requests.len() < self.batch_size && terminal_gathers < self.batch_size {
                match zero_step_gather(tree, oracle, self.weights, self.use_value, self.fpu_mode) {
                    Some(request) => {
                        requests.push(request);
                    }
                    None => {
                        terminal_gathers += 1;
                    }
                }
            }

            // ask the network to evaluate
            let boards = requests.iter().map(|r| &r.board).collect_vec();
            let evals = network.evaluate_batch(&boards);

            // add all evaluations back to the tree
            for (req, eval) in zip_eq_exact(requests, evals) {
                zero_step_apply(tree, req.respond(eval));
            }
        }
    }
}

pub struct ZeroBot<B: Board, N: Network<B>, O: Oracle<B>> {
    network: N,
    oracle: O,
    settings: ZeroSettings,
    visits: u64,
    ph: PhantomData<B>,
}

impl<B: Board, N: Network<B>, O: Oracle<B>> ZeroBot<B, N, O> {
    pub fn new(network: N, settings: ZeroSettings, oracle: O, visits: u64) -> Self {
        assert!(visits > 0, "Need at least one visit to pick the best move");
        ZeroBot {
            network,
            settings,
            oracle,
            visits,
            ph: PhantomData,
        }
    }

    pub fn build_tree(&mut self, board: &B) -> Tree<B> {
        let visits = self.visits;
        let stop = |tree: &Tree<B>| tree.root_visits() >= visits;
        let tree = self.settings.build_tree(board, &mut self.network, &self.oracle, stop);
        tree
    }
}

impl<B: Board, N: Network<B>, O: Oracle<B>> Bot<B> for ZeroBot<B, N, O> {
    fn select_move(&mut self, board: &B) -> B::Move {
        assert!(!board.is_done());

        if let Some(eval) = self.oracle.evaluate(board) {
            return eval.best_move.unwrap();
        }

        self.build_tree(board).best_move().unwrap()
    }
}

impl<B: Board, N: Network<B>, O: Oracle<B>> Debug for ZeroBot<B, N, O> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeroBot")
            .field("settings", &self.settings)
            .field("visits", &self.visits)
            .field("network", &self.network)
            .field("oracle", &self.oracle)
            .finish()
    }
}
