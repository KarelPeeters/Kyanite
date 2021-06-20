use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam::channel::{Sender, SendError};
use itertools::Itertools;
use itertools::izip;
use rand::{Rng, thread_rng};
use sttt::board::Board;
use tch::Device;

use crate::mcts_zero::{KeepResult, Request, Response, RunResult, Tree, zero_build_tree, ZeroState};
use crate::network::google_torch::GoogleTorchNetwork;
use crate::network::Network;
use crate::selfplay::{Generator, Message, MoveSelector, Position, Simulation};

#[derive(Debug)]
pub struct ZeroGenerator {
    // performance settings
    pub devices: Vec<Device>,
    pub threads_per_device: usize,
    pub batch_size: usize,

    // settings that effect the generated games
    pub network_path: String,
    pub iterations: u64,
    pub exploration_weight: f32,
}

type Net = GoogleTorchNetwork;

impl ZeroGenerator {
    fn load_network(&self, device: Device) -> Net {
        Net::load(&self.network_path, device)
    }

    fn new_zero(&self, tree: Tree) -> ZeroState {
        ZeroState::new(tree, self.iterations, self.exploration_weight)
    }
}

impl Generator for ZeroGenerator {
    type Init = Tree;
    type ThreadInit = Device;

    fn initialize(&self) -> Self::Init {
        let mut network = self.load_network(Device::Cpu);
        zero_build_tree(&Board::new(), self.iterations, self.exploration_weight, &mut network)
    }

    fn thread_initialize(&self) -> Vec<Self::ThreadInit> {
        self.devices.repeat(self.threads_per_device)
    }

    fn thread_main(
        &self,
        move_selector: &MoveSelector,
        root_tree: &Tree,
        device: Device,
        request_stop: &AtomicBool,
        sender: &Sender<Message>,
    ) -> Result<(), SendError<Message>> {
        let batch_size = self.batch_size;
        let mut network = self.load_network(device);
        let mut rng = thread_rng();

        let mut states = vec![GameState::new(self.new_zero(root_tree.clone())); batch_size];
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
                    root_tree,
                    response,
                    sender,
                )?;
                total_move_count += move_count;
                requests.push(request);
            }

            //pass requests to network
            let boards = requests.iter().map(|r| r.board.clone()).collect_vec();
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
        settings: &ZeroGenerator,
        root_tree: &Tree,
        response: Option<Response>,
        sender: &Sender<Message>,
    ) -> Result<(u64, Request), SendError<Message>> {
        let mut response = response;
        let mut move_count = 0;

        loop {
            let result = self.zero.run_until_result(response.take());

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
                        value: tree[0].value(),
                        policy,
                    });
                    move_count += 1;

                    //keep the tree for the picked move
                    match tree.keep_move(picked_move) {
                        KeepResult::Tree(tree) => {
                            //continue playing this game
                            self.zero = settings.new_zero(tree)
                        }
                        KeepResult::Done(won_by) => {
                            //record this game
                            let simulation = Simulation { won_by, positions: std::mem::take(&mut self.positions) };
                            sender.send(Message::Simulation(simulation))?;

                            //start a new game
                            *self = GameState::new(settings.new_zero(root_tree.clone()));
                        }
                    }
                }
            }
        }
    }
}

