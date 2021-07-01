use std::fmt::Debug;

use crossbeam::channel::Sender;
use itertools::Itertools;
use itertools::izip;
use rand::{Rng, thread_rng};
use rand_distr::Dirichlet;
use sttt::board::Board;

use crate::mcts_zero::{KeepResult, Request, Response, RunResult, Tree, ZeroSettings, ZeroState};
use crate::network::Network;
use crate::selfplay::{Generator, Message, MoveSelector, Position, Simulation, StartGameCounter};

#[derive(Debug)]
pub struct ZeroGeneratorSettings<S: NetworkSettings> {
    // performance settings
    pub batch_size: usize,

    // settings that effect the generated games
    //TODO add random moves with less visits to reduce value overfitting
    pub network: S,
    pub iterations: u64,
    pub zero_settings: ZeroSettings,

    pub keep_tree: bool,
    pub dirichlet_alpha: f32,
    pub dirichlet_eps: f32,
}

pub trait NetworkSettings: Debug + Sync {
    type ThreadParam: Send;
    type Network: Network;

    fn load_network(&self, param: Self::ThreadParam) -> Self::Network;

    fn thread_params(&self) -> Vec<Self::ThreadParam>;
}

#[cfg(feature = "torch")]
pub mod settings_torch {
    use tch::Device;

    use crate::network::google_torch::GoogleTorchNetwork;
    use crate::selfplay::generate_zero::NetworkSettings;

    #[derive(Debug)]
    pub struct GoogleTorchSettings {
        pub path: String,
        pub devices: Vec<Device>,
        pub threads_per_device: usize,
    }

    impl NetworkSettings for GoogleTorchSettings {
        type ThreadParam = Device;
        type Network = GoogleTorchNetwork;

        fn load_network(&self, init: Self::ThreadParam) -> Self::Network {
            GoogleTorchNetwork::load(&self.path, init)
        }

        fn thread_params(&self) -> Vec<Self::ThreadParam> {
            self.devices.repeat(self.threads_per_device)
        }
    }
}

#[cfg(feature = "onnx")]
pub mod settings_onnx {
    use crate::network::google_onnx::GoogleOnnxNetwork;
    use crate::selfplay::generate_zero::NetworkSettings;

    #[derive(Debug)]
    pub struct GoogleOnnxSettings {
        pub path: String,
        pub num_threads: usize,
    }

    impl NetworkSettings for GoogleOnnxSettings {
        type ThreadParam = ();
        type Network = GoogleOnnxNetwork;

        fn load_network(&self, _: ()) -> Self::Network {
            GoogleOnnxNetwork::load(&self.path)
        }

        fn thread_params(&self) -> Vec<()> {
            vec![(); self.num_threads]
        }
    }
}

impl<S: NetworkSettings> ZeroGeneratorSettings<S> {
    fn new_zero(&self, tree: Tree) -> ZeroState {
        ZeroState::new(tree, self.iterations, self.zero_settings)
    }

    fn new_zero_root(&self) -> ZeroState {
        self.new_zero(Tree::new(Board::new()))
    }

    fn add_dirichlet_noise(&self, tree: &mut Tree, rng: &mut impl Rng) {
        let children = tree[0].children
            .expect("root node has no children yet, it must have been visited at least once");

        if children.length > 1 {
            let distr = Dirichlet::new_with_size(self.dirichlet_alpha, children.length as usize).unwrap();
            let noise = rng.sample(distr);

            for (child, n) in izip!(children, noise) {
                tree[child].policy.0 += n
            }
        }
    }
}

impl<S: NetworkSettings> Generator for ZeroGeneratorSettings<S> {
    type ThreadParam = S::ThreadParam;

    fn thread_params(&self) -> Vec<Self::ThreadParam> {
        self.network.thread_params()
    }

    fn thread_main(
        &self,
        move_selector: &MoveSelector,
        thread_param: S::ThreadParam,
        start_counter: &StartGameCounter,
        sender: &Sender<Message>,
    ) {
        let mut network = self.network.load_network(thread_param);
        let mut rng = thread_rng();

        let mut games: Vec<GameState> = vec![];
        let mut requests: Vec<Request> = vec![];
        let mut responses: Vec<Response> = vec![];

        loop {
            let mut total_move_count = 0;

            // run all existing games and collect the requests
            assert_eq!(games.len(), responses.len());
            assert!(requests.is_empty());
            let mut kept_games = vec![];

            for (mut game, response) in izip!(games.drain(..), responses.drain(..)) {
                let (request, move_count) = game.run_until_request(&mut rng, move_selector, self, Some(response), sender);
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
                let mut game = GameState::new(self.new_zero_root());
                let (request, move_count) = game.run_until_request(&mut rng, move_selector, self, None, sender);

                total_move_count += move_count;
                let request = request.expect("The first run of a gamestate should always returns a request");

                games.push(game);
                requests.push(request);
            }

            let request_count = requests.len();
            if request_count == 0 { break; }

            //pass requests to network
            let boards = requests.iter().map(|r| r.board()).collect_vec();
            let mut evaluations = network.evaluate_batch(&boards);

            assert!(responses.is_empty());
            responses.extend(
                izip!(requests.drain(..), evaluations.drain(..))
                    .map(|(request, evaluation)| Response { request, evaluation })
            );

            //send the number of evaluations that happened
            sender.send(Message::Counter { evals: request_count as u64, moves: total_move_count }).unwrap();
        }
    }
}

// The state kept while generating a new game.
#[derive(Debug)]
struct GameState {
    zero: ZeroState,
    needs_dirichlet: bool,
    positions: Vec<Position>,
}

impl GameState {
    fn new(zero: ZeroState) -> Self {
        GameState { zero, needs_dirichlet: true, positions: vec![] }
    }

    /// Run this game until either:
    /// * a request is made, in which case that request is returned
    /// * the game is done, in which case `None` is returned
    fn run_until_request(
        &mut self,
        rng: &mut impl Rng,
        move_selector: &MoveSelector,
        settings: &ZeroGeneratorSettings<impl NetworkSettings>,
        response: Option<Response>,
        sender: &Sender<Message>,
    ) -> (Option<Request>, u64) {
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
                    let picked_index = move_selector.select(tree.root_board().count_tiles(), &policy, rng);
                    let picked_child = tree[0].children.unwrap().get(picked_index);
                    let picked_move = tree[picked_child].coord.unwrap();

                    //store this position
                    self.positions.push(Position {
                        board: self.zero.tree.root_board().clone(),
                        value: tree.value(),
                        policy,
                    });
                    move_count += 1;

                    //keep the tree for the picked move
                    match tree.keep_move(picked_move) {
                        KeepResult::Tree(next_tree) => {
                            //continue playing this game, either by keeping part of the tree or starting a new one on the next board
                            if settings.keep_tree {
                                self.zero = settings.new_zero(next_tree)
                            } else {
                                self.zero = settings.new_zero(Tree::new(next_tree.root_board().clone()))
                            }
                        }
                        KeepResult::Done(won_by) => {
                            //record this game
                            let simulation = Simulation { won_by, positions: std::mem::take(&mut self.positions) };
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

