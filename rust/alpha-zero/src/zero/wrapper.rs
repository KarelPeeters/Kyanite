use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;

use board_game::ai::Bot;
use board_game::board::Board;
use itertools::Itertools;

use crate::network::Network;
use crate::oracle::Oracle;
use crate::util::zip_eq_exact;
use crate::zero::step::{zero_step_apply, zero_step_gather};
use crate::zero::tree::Tree;

#[derive(Debug, Copy, Clone)]
pub struct ZeroSettings {
    pub batch_size: usize,
    pub exploration_weight: f32,
    pub use_value: bool,
}

impl ZeroSettings {
    pub fn new(batch_size: usize, exploration_weight: f32, use_value: bool) -> Self {
        ZeroSettings { batch_size, exploration_weight, use_value }
    }
}

impl ZeroSettings {
    /// Construct a new tree from scratch on the given board.
    pub fn build_tree<B: Board>(
        self,
        root_board: &B,
        network: &mut impl Network<B>,
        oracle: &impl Oracle<B>,
        stop: impl Fn(&Tree<B>) -> bool,
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
            // TODO what about resuming after we've stopped? we don't want to drop the requests!
            //   moving the check to once per outer loop is correct but very coarse
            if stop(tree) { break 'outer; }

            let mut requests = vec![];

            // collect enough requests to fill the batch
            // TODO what about when we have explored the entire tree and are left with a half-filled batch?
            while requests.len() < self.batch_size {
                let request = zero_step_gather(tree, oracle, self.exploration_weight, self.use_value);
                if let Some(request) = request {
                    requests.push(request);
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
        ZeroBot { network, settings, oracle, visits, ph: PhantomData }
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

        self.build_tree(board).best_move()
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
