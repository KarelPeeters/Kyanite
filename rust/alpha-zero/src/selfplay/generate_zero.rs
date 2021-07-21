use std::fmt::Debug;
use std::marker::PhantomData;

use crossbeam::channel::Sender;
use itertools::Itertools;
use itertools::izip;
use rand::{Rng, thread_rng};
use rand_distr::Dirichlet;
use sttt::board::{Board, Outcome};

use crate::network::Network;
use crate::selfplay::{Generator, Message, MoveSelector, Position, Simulation, StartGameCounter};
use crate::zero::{KeepResult, Request, Response, RunResult, Tree, ZeroEvaluation, ZeroSettings, ZeroState};

#[derive(Debug)]
pub struct ZeroGeneratorSettings<B: Board, S: NetworkSettings<B>> {
    // performance settings
    pub batch_size: usize,

    // settings that effect the generated games
    pub network: S,
    pub zero_settings: ZeroSettings,

    pub full_search_prob: f64,
    pub full_iterations: u64,
    pub part_iterations: u64,

    pub keep_tree: bool,
    pub dirichlet_alpha: f32,
    pub dirichlet_eps: f32,

    pub max_game_length: u32,

    pub ph: PhantomData<B>,
}

pub trait NetworkSettings<B: Board>: Debug + Sync {
    type ThreadParam: Send;
    type Network: Network<B>;

    fn load_network(&self, param: Self::ThreadParam) -> Self::Network;

    fn thread_params(&self) -> Vec<Self::ThreadParam>;
}

impl<B: Board, S: NetworkSettings<B>> ZeroGeneratorSettings<B, S> {
    fn new_zero(&self, tree: Tree<B>, rng: &mut impl Rng) -> ZeroState<B> {
        let iterations = if rng.gen_bool(self.full_search_prob) {
            self.full_iterations
        } else {
            self.part_iterations
        };

        ZeroState::new(tree, iterations, self.zero_settings)
    }

    fn new_zero_root(&self, start_board: &B, rng: &mut impl Rng) -> ZeroState<B> {
        self.new_zero(Tree::new(start_board.clone()), rng)
    }

    fn add_dirichlet_noise(&self, tree: &mut Tree<B>, rng: &mut impl Rng) {
        let children = tree[0].children
            .expect("root node has no children yet, it must have been visited at least once");

        if children.length > 1 {
            let distr = Dirichlet::new_with_size(self.dirichlet_alpha, children.length as usize).unwrap();
            let noise = rng.sample(distr);

            for (child, n) in izip!(children, noise) {
                tree[child].net_policy += n
            }
        }
    }
}

impl<B: Board, S: NetworkSettings<B>> Generator<B> for ZeroGeneratorSettings<B, S> {
    type ThreadParam = S::ThreadParam;

    fn thread_params(&self) -> Vec<Self::ThreadParam> {
        self.network.thread_params()
    }

    fn thread_main(
        &self,
        start_board: &B,
        move_selector: MoveSelector,
        thread_param: S::ThreadParam,
        start_counter: &StartGameCounter,
        sender: Sender<Message<B>>,
    ) {
        //TODO combine threads as soon as the sum of batch sizes gets low enough
        // alternative: switch to continuously generating games with changing network
        // but then training may be too slow since the GPUs are used a lot?

        let mut network = self.network.load_network(thread_param);
        let rng = &mut thread_rng();

        let mut games: Vec<GameState<B>> = vec![];
        let mut requests: Vec<Request<B>> = vec![];
        let mut responses: Vec<Response<B>> = vec![];

        loop {
            let mut total_move_count = 0;

            // run all existing games and collect the requests
            assert_eq!(games.len(), responses.len());
            assert!(requests.is_empty());
            let mut kept_games = vec![];

            for (mut game, response) in izip!(games.drain(..), responses.drain(..)) {
                let (request, move_count) = game.run_until_request(rng, move_selector, self, Some(response), &sender);
                total_move_count += move_count;

                if let Some(request) = request {
                    // this game is not done, keep it and its request
                    kept_games.push(game);
                    requests.push(request);
                }
            }

            games = kept_games;

            // create new games until we have enough and run them once
            let new_game_count = start_counter.request_up_to((self.batch_size - games.len()) as u64);
            for _ in 0..new_game_count {
                let mut game = GameState::new(self.new_zero_root(start_board, rng));
                let (request, move_count) = game.run_until_request(rng, move_selector, self, None, &sender);

                total_move_count += move_count;
                let request = request.expect("The first run of a gamestate should always returns a request");

                games.push(game);
                requests.push(request);
            }

            let request_count = requests.len();
            if request_count == 0 { break; }

            //pass requests to network
            assert!(responses.is_empty());
            responses = network.evaluate_batch_requests(&requests);
            requests.clear();

            //send the number of evaluations that happened
            sender.send(Message::Counter { evals: request_count as u64, moves: total_move_count }).unwrap();
        }
    }
}

// The state kept while generating a new game.
#[derive(Debug)]
struct GameState<B: Board> {
    zero: ZeroState<B>,
    needs_dirichlet: bool,
    positions: Vec<Position<B>>,
}

impl<B: Board> GameState<B> {
    fn new(zero: ZeroState<B>) -> Self {
        GameState { zero, needs_dirichlet: true, positions: vec![] }
    }

    /// Run this game until either:
    /// * a request is made, in which case that request is returned
    /// * the game is done, in which case `None` is returned
    fn run_until_request(
        &mut self,
        rng: &mut impl Rng,
        move_selector: MoveSelector,
        settings: &ZeroGeneratorSettings<B, impl NetworkSettings<B>>,
        response: Option<Response<B>>,
        sender: &Sender<Message<B>>,
    ) -> (Option<Request<B>>, u64) {
        let mut response = response;
        let mut move_count = 0;

        loop {
            let had_response = response.is_some();
            let result = self.zero.run_until_result(response.take(), rng);

            if had_response && self.needs_dirichlet {
                // at this point we're sure that the tree has at least one visit, se we can add the noise
                settings.add_dirichlet_noise(&mut self.zero.tree, rng);
                self.needs_dirichlet = false;
            }

            match result {
                RunResult::Request(request) => {
                    return (Some(request), move_count);
                }
                RunResult::Done => {
                    let tree = &self.zero.tree;
                    let policy = tree.policy().collect_vec();

                    //pick a move to play
                    let picked_index = move_selector.select(self.positions.len() as u32, policy.iter().copied(), rng);
                    let picked_child = tree[0].children.unwrap().get(picked_index);
                    let picked_move = tree[picked_child].last_move.unwrap();

                    //store this position
                    let iterations = self.zero.target_iterations;
                    assert!(iterations == settings.full_iterations || iterations == settings.part_iterations);
                    let should_store = iterations == settings.full_iterations;

                    self.positions.push(Position {
                        board: self.zero.tree.root_board().clone(),
                        should_store,
                        evaluation: ZeroEvaluation {
                            wdl: tree.wdl(),
                            policy,
                        },
                    });
                    move_count += 1;

                    // decide whether to continue this game
                    let result = if self.positions.len() as u32 >= settings.max_game_length {
                        KeepResult::Done(Outcome::Draw)
                    } else {
                        tree.keep_move(picked_move)
                    };

                    match result {
                        KeepResult::Tree(next_tree) => {
                            //continue playing this game, either by keeping part of the tree or starting a new one on the next board
                            if settings.keep_tree {
                                self.zero = settings.new_zero(next_tree, rng)
                            } else {
                                self.zero = settings.new_zero(Tree::new(next_tree.root_board().clone()), rng)
                            }
                        }
                        KeepResult::Done(outcome) => {
                            //record this game
                            let simulation = Simulation { outcome, positions: std::mem::take(&mut self.positions) };
                            sender.send(Message::Simulation(simulation)).unwrap();

                            //report that this game is done
                            return (None, move_count);
                        }
                    }
                }
            }
        }
    }
}

