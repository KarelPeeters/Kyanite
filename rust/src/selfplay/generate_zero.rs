use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam::channel::{Sender, SendError};
use itertools::Itertools;
use itertools::izip;
use rand::{Rng, thread_rng};
use sttt::board::Board;

use crate::mcts_zero::{KeepResult, Request, Response, RunResult, Tree, ZeroSettings, ZeroState};
use crate::network::Network;
use crate::selfplay::{Generator, Message, MoveSelector, Position, Simulation};

#[derive(Debug)]
pub struct ZeroGeneratorSettings<S: NetworkSettings> {
    // performance settings
    pub batch_size: usize,

    // settings that effect the generated games
    //TODO add random moves with less visits to reduce value overfitting
    pub network: S,
    pub iterations: u64,
    pub zero_settings: ZeroSettings,
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
        request_stop: &AtomicBool,
        sender: &Sender<Message>,
    ) -> Result<(), SendError<Message>> {
        let batch_size = self.batch_size;
        let mut network = self.network.load_network(thread_param);
        let mut rng = thread_rng();

        let mut states = vec![GameState::new(self.new_zero_root()); batch_size];
        let mut responses = (0..batch_size).map(|_| None).collect_vec();
        let mut requests = Vec::new();

        loop {
            //early exit
            if request_stop.load(Ordering::SeqCst) { return Ok(()); };

            let mut total_move_count = 0;

            //advance all games and collect the next batch of requests
            for i in 0..batch_size {
                let state = &mut states[i];

                let response = responses[i].take();
                let (move_count, request) = state.run_until_request(
                    &mut rng,
                    move_selector,
                    self,
                    response,
                    sender,
                )?;
                total_move_count += move_count;
                requests.push(request);
            }

            //pass requests to network
            let boards = requests.iter().map(|r| r.board()).collect_vec();
            let mut evaluations = network.evaluate_batch(&boards);

            //construct responses
            let iter = izip!(&mut responses, requests.drain(..), evaluations.drain(..));
            for (response, request, evaluation) in iter {
                assert!(response.is_none());
                *response = Some(Response { request, evaluation });
            }

            //send the number of evaluations that happened
            sender.send(Message::Counter { evals: batch_size as u64, moves: total_move_count })?;
        }
    }
}

// The state kept while generating a new game.
#[derive(Debug, Clone)]
struct GameState {
    zero: ZeroState,
    positions: Vec<Position>,
}

impl GameState {
    fn new(zero: ZeroState) -> Self {
        GameState { zero, positions: vec![] }
    }

    fn run_until_request(
        &mut self,
        rng: &mut impl Rng,
        move_selector: &MoveSelector,
        settings: &ZeroGeneratorSettings<impl NetworkSettings>,
        response: Option<Response>,
        sender: &Sender<Message>,
    ) -> Result<(u64, Request), SendError<Message>> {
        let mut response = response;
        let mut move_count = 0;

        loop {
            let result = self.zero.run_until_result(response.take(), rng);

            match result {
                RunResult::Request(request) =>
                    return Ok((move_count, request)),
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
                        KeepResult::Tree(tree) => {
                            //TODO maybe remove this when dirichlet noise is added since it's not equivalent any more?
                            //  or still do this but add the dirichlet anyway? could be interesting
                            //  this will become a lot less relevant once we add random moves with less iterations anyway
                            //continue playing this game
                            self.zero = settings.new_zero(tree)
                        }
                        KeepResult::Done(won_by) => {
                            //record this game
                            let simulation = Simulation { won_by, positions: std::mem::take(&mut self.positions) };
                            sender.send(Message::Simulation(simulation))?;

                            //start a new game
                            *self = GameState::new(settings.new_zero_root());
                        }
                    }
                }
            }
        }
    }
}

