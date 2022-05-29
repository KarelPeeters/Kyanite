use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;

use async_trait::async_trait;
use board_game::ai::Bot;
use board_game::board::Board;
use flume::RecvError;
use futures::executor::block_on;
use itertools::Itertools;

use kz_util::sequence::zip_eq_exact;

use crate::bot::AsyncBot;
use crate::network::{EvalClient, Network};
use crate::network::job_channel::{Job, job_pair};
use crate::zero::node::UctWeights;
use crate::zero::step::{FpuMode, zero_step_apply, zero_step_gather};
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
    /// Utility wrapper around [Self::build_tree_async] that spawns a temporary thread pool.
    pub fn build_tree<B: Board>(
        self,
        root_board: &B,
        network: &mut impl Network<B>,
        stop: impl FnMut(&Tree<B>) -> bool + Send,
    ) -> Tree<B> {
        let mut tree = Tree::new(root_board.clone());
        self.expand_tree(&mut tree, network, stop);
        tree
    }

    /// Construct a new tree from scratch on the given board.
    pub async fn build_tree_async<B: Board>(
        self,
        root_board: &B,
        eval_client: &EvalClient<B>,
        stop: impl FnMut(&Tree<B>) -> bool,
    ) -> Tree<B> {
        let mut tree = Tree::new(root_board.clone());
        self.expand_tree_async(&mut tree, eval_client, stop).await;
        tree
    }

    /// Utility wrapper around [Self::expand_tree] that spawns a temporary thread pool.
    pub fn expand_tree<B: Board>(
        self,
        tree: &mut Tree<B>,
        network: &mut impl Network<B>,
        stop: impl FnMut(&Tree<B>) -> bool + Send,
    ) {
        let (client, server) = job_pair(1);

        crossbeam::scope(|s| {
            // build the tree itself in a new thread
            s.spawn(|_| {
                block_on(async move {
                    self.expand_tree_async(tree, &client, stop).await;
                    drop(client);
                })
            });

            // use this thread for network inference
            loop {
                match server.receiver().recv() {
                    Ok(Job { x, sender }) => {
                        sender.send(network.evaluate_batch(&x)).unwrap();
                    }
                    Err(RecvError::Disconnected) => break,
                }
            }

            // implicitly join async thread
        })
            .unwrap();
    }

    // Continue expanding an existing tree.
    pub async fn expand_tree_async<B: Board>(
        self,
        tree: &mut Tree<B>,
        eval_client: &EvalClient<B>,
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
                match zero_step_gather(tree, self.weights, self.use_value, self.fpu_mode) {
                    Some(request) => {
                        requests.push(request);
                    }
                    None => {
                        terminal_gathers += 1;
                    }
                }
            }

            // ask the network to evaluate
            let boards = requests.iter().map(|r| r.board.clone()).collect_vec();
            let evals = eval_client.map_async(boards).await;

            // add all evaluations back to the tree
            for (req, eval) in zip_eq_exact(requests, evals) {
                zero_step_apply(tree, req.respond(eval));
            }
        }
    }
}

pub struct ZeroBot<B: Board, N: Network<B>> {
    network: N,
    settings: ZeroSettings,
    visits: u64,
    ph: PhantomData<B>,
}

impl<B: Board, N: Network<B>> ZeroBot<B, N> {
    pub fn new(network: N, settings: ZeroSettings, visits: u64) -> Self {
        assert!(visits > 0, "Need at least one visit to pick the best move");
        ZeroBot {
            network,
            settings,
            visits,
            ph: PhantomData,
        }
    }

    pub fn build_tree(&mut self, board: &B) -> Tree<B> {
        let visits = self.visits;
        let stop = |tree: &Tree<B>| tree.root_visits() >= visits;
        let tree = self.settings.build_tree(board, &mut self.network, stop);
        tree
    }
}

impl<B: Board, N: Network<B>> Bot<B> for ZeroBot<B, N> {
    fn select_move(&mut self, board: &B) -> B::Move {
        assert!(!board.is_done());
        self.build_tree(board).best_move().unwrap()
    }
}

impl<B: Board, N: Network<B>> Debug for ZeroBot<B, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeroBot")
            .field("settings", &self.settings)
            .field("visits", &self.visits)
            .field("network", &self.network)
            .finish()
    }
}

pub struct AsyncZeroBot<B: Board> {
    eval_client: EvalClient<B>,
    settings: ZeroSettings,
    visits: u64,
}

impl<B: Board> AsyncZeroBot<B> {
    pub fn new(eval_client: EvalClient<B>, settings: ZeroSettings, visits: u64) -> Self {
        AsyncZeroBot {
            eval_client,
            settings,
            visits,
        }
    }

    pub async fn build_tree(&mut self, board: &B) -> Tree<B> {
        let visits = self.visits;
        let stop = |tree: &Tree<B>| tree.root_visits() >= visits;
        let tree = self.settings.build_tree_async(board, &self.eval_client, stop).await;
        tree
    }
}

#[async_trait]
impl<B: Board> AsyncBot<B> for AsyncZeroBot<B> {
    async fn select_move(&mut self, board: &B) -> B::Move {
        assert!(!board.is_done());
        self.build_tree(board).await.best_move().unwrap()
    }
}

impl<B: Board> Debug for AsyncZeroBot<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncZeroBot")
            .field("settings", &self.settings)
            .field("visits", &self.visits)
            .field("eval_client", &self.eval_client)
            .finish()
    }
}
