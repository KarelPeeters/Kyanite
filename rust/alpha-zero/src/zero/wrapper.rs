use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;

use board_game::ai::Bot;
use board_game::board::Board;
use itertools::Itertools;

use crate::network::Network;
use crate::util::zip_eq_exact;
use crate::zero::step::{zero_step_apply, zero_step_gather};
use crate::zero::stop_cond::StopCondition;
use crate::zero::tree::Tree;

#[derive(Debug, Copy, Clone)]
pub struct ZeroSettings {
    batch_size: usize,
    exploration_weight: f32,
}

impl ZeroSettings {
    pub fn new(batch_size: usize, exploration_weight: f32) -> Self {
        ZeroSettings { batch_size, exploration_weight }
    }
}

impl ZeroSettings {
    /// Construct a new tree from scratch on the given board.
    pub fn build_tree<B: Board>(
        self,
        root_board: &B,
        network: &mut impl Network<B>,
        stop: &impl StopCondition<B>,
    ) -> Tree<B> {
        let mut tree = Tree::new(root_board.clone());

        'outer: loop {
            let mut requests = vec![];

            // collect enough requests to fill the batch
            // TODO what about when we have explored the entire tree and are left with a half-filled batch?
            while requests.len() < self.batch_size {
                if stop.should_stop(&tree) { break 'outer; }

                let request = zero_step_gather(&mut tree, self.exploration_weight);
                if let Some(request) = request {
                    requests.push(request);
                }
            }

            // ask the network to evaluate
            let boards = requests.iter().map(|r| &r.board).collect_vec();
            let evals = network.evaluate_batch(&boards);

            // either add all evaluations or none
            if stop.should_stop(&tree) { break 'outer; }

            // add all evaluations back to the tree
            for (req, eval) in zip_eq_exact(requests, evals) {
                zero_step_apply(&mut tree, req.respond(eval));
            }
        }

        tree
    }
}

pub struct ZeroBot<B: Board, N: Network<B>, S: StopCondition<B>> {
    network: N,
    settings: ZeroSettings,
    stop: S,
    ph: PhantomData<B>,
}

impl<B: Board, N: Network<B>, S: StopCondition<B>> ZeroBot<B, N, S> {
    pub fn new(network: N, settings: ZeroSettings, stop: S) -> Self {
        ZeroBot { network, settings, stop, ph: PhantomData }
    }
}

impl<B: Board, N: Network<B>, S: StopCondition<B>> Bot<B> for ZeroBot<B, N, S> {
    fn select_move(&mut self, board: &B) -> B::Move {
        let tree = self.settings.build_tree(board, &mut self.network, &self.stop);
        tree.best_move()
    }
}

impl<B: Board, N: Network<B>, S: StopCondition<B>> Debug for ZeroBot<B, N, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeroBot")
            .field("settings", &self.settings)
            .field("stop", &self.stop.debug())
            .field("network", &self.network)
            .finish()
    }
}
