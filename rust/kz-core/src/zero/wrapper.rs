use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;

use async_trait::async_trait;
use board_game::ai::Bot;
use board_game::board::Board;
use flume::RecvError;
use futures::executor::block_on;
use itertools::Itertools;
use rand::Rng;

use kz_util::sequence::zip_eq_exact;

use crate::bot::AsyncBot;
use crate::network::common::policy_softmax_temperature_in_place;
use crate::network::job_channel::{job_pair, Job};
use crate::network::{EvalClient, Network};
use crate::zero::node::UctWeights;
use crate::zero::step::{zero_step_apply, zero_step_gather, FpuMode};
use crate::zero::tree::Tree;

#[derive(Debug, Copy, Clone)]
pub struct ZeroSettings {
    pub batch_size: usize,
    pub weights: UctWeights,
    pub use_value: bool,
    pub fpu_root: FpuMode,
    pub fpu_child: FpuMode,
    pub policy_temperature: f32,
}

impl ZeroSettings {
    pub fn simple(batch_size: usize, weights: UctWeights, fpu: FpuMode) -> ZeroSettings {
        ZeroSettings {
            batch_size,
            weights,
            use_value: false,
            fpu_root: fpu,
            fpu_child: fpu,
            policy_temperature: 1.0,
        }
    }

    pub fn new(
        batch_size: usize,
        weights: UctWeights,
        use_value: bool,
        fpu_root: FpuMode,
        fpu_child: FpuMode,
        policy_temperature: f32,
    ) -> Self {
        Self {
            batch_size,
            weights,
            use_value,
            fpu_root,
            fpu_child,
            policy_temperature,
        }
    }

    pub fn fpu_mode(&self, is_root: bool) -> FpuMode {
        if is_root {
            self.fpu_root
        } else {
            self.fpu_child
        }
    }
}

impl ZeroSettings {
    /// Utility wrapper around [Self::build_tree_async] that spawns a temporary thread pool.
    pub fn build_tree<B: Board>(
        self,
        root_board: &B,
        network: &mut impl Network<B>,
        rng: &mut (impl Rng + Send),
        stop: impl FnMut(&Tree<B>) -> bool + Send,
    ) -> Tree<B> {
        let mut tree = Tree::new(root_board.clone());
        self.expand_tree(&mut tree, network, rng, stop);
        tree
    }

    /// Construct a new tree from scratch on the given board.
    pub async fn build_tree_async<B: Board>(
        self,
        root_board: &B,
        eval_client: &EvalClient<B>,
        rng: &mut impl Rng,
        stop: impl FnMut(&Tree<B>) -> bool,
    ) -> Tree<B> {
        let mut tree = Tree::new(root_board.clone());
        self.expand_tree_async(&mut tree, eval_client, rng, stop).await;
        tree
    }

    /// Utility wrapper around [Self::expand_tree] that spawns a temporary thread pool.
    pub fn expand_tree<B: Board>(
        self,
        tree: &mut Tree<B>,
        network: &mut impl Network<B>,
        rng: &mut (impl Rng + Send),
        stop: impl FnMut(&Tree<B>) -> bool + Send,
    ) {
        let (client, server) = job_pair(1);

        crossbeam::scope(|s| {
            // build the tree itself in a new thread
            s.spawn(|_| {
                block_on(async move {
                    self.expand_tree_async(tree, &client, rng, stop).await;
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
        rng: &mut impl Rng,
        mut stop: impl FnMut(&Tree<B>) -> bool,
    ) {
        while !stop(tree) {
            // collect requests until the batch is full or we repeatedly fail to find new positions to evaluate
            let mut requests = vec![];
            let mut terminal_gathers = 0;

            while requests.len() < self.batch_size && terminal_gathers < self.batch_size {
                match zero_step_gather(tree, self.weights, self.use_value, self.fpu_root, self.fpu_child, rng) {
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
            for (req, mut eval) in zip_eq_exact(requests, evals) {
                policy_softmax_temperature_in_place(eval.policy.to_mut(), self.policy_temperature);
                zero_step_apply(tree, req.respond(eval));
            }
        }
    }
}

pub struct ZeroBot<B: Board, N: Network<B>, R: Rng + Send> {
    network: N,
    settings: ZeroSettings,
    visits: u64,
    rng: R,
    ph: PhantomData<B>,
}

impl<B: Board, N: Network<B>, R: Rng + Send> ZeroBot<B, N, R> {
    pub fn new(network: N, settings: ZeroSettings, visits: u64, rng: R) -> Self {
        assert!(visits > 0, "Need at least one visit to pick the best move");
        ZeroBot {
            network,
            settings,
            visits,
            rng,
            ph: PhantomData,
        }
    }

    pub fn build_tree(&mut self, board: &B) -> Tree<B> {
        let visits = self.visits;
        let stop = |tree: &Tree<B>| tree.root_visits() >= visits;
        let tree = self.settings.build_tree(board, &mut self.network, &mut self.rng, stop);
        tree
    }
}

impl<B: Board, N: Network<B>, R: Rng + Send> Bot<B> for ZeroBot<B, N, R> {
    fn select_move(&mut self, board: &B) -> B::Move {
        assert!(!board.is_done());
        self.build_tree(board).best_move().unwrap()
    }
}

impl<B: Board, N: Network<B>, R: Rng + Send> Debug for ZeroBot<B, N, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeroBot")
            .field("settings", &self.settings)
            .field("visits", &self.visits)
            .field("network", &self.network)
            .finish()
    }
}

pub struct AsyncZeroBot<B: Board, R: Rng + Send> {
    eval_client: EvalClient<B>,
    settings: ZeroSettings,
    visits: u64,
    rng: R,
}

impl<B: Board, R: Rng + Send> AsyncZeroBot<B, R> {
    pub fn new(eval_client: EvalClient<B>, settings: ZeroSettings, visits: u64, rng: R) -> Self {
        AsyncZeroBot {
            eval_client,
            settings,
            visits,
            rng,
        }
    }

    pub async fn build_tree(&mut self, board: &B) -> Tree<B> {
        let visits = self.visits;
        let stop = |tree: &Tree<B>| tree.root_visits() >= visits;
        let tree = self
            .settings
            .build_tree_async(board, &self.eval_client, &mut self.rng, stop)
            .await;
        tree
    }
}

#[async_trait]
impl<B: Board, R: Rng + Send> AsyncBot<B> for AsyncZeroBot<B, R> {
    async fn select_move(&mut self, board: &B) -> B::Move {
        assert!(!board.is_done());
        self.build_tree(board).await.best_move().unwrap()
    }
}

impl<B: Board, R: Rng + Send> Debug for AsyncZeroBot<B, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncZeroBot")
            .field("settings", &self.settings)
            .field("visits", &self.visits)
            .field("eval_client", &self.eval_client)
            .finish()
    }
}
