use tch::{Device, Cuda};
use sttt::board::{Board, Player};
use itertools::Itertools;
use std::time::Instant;
use crossbeam::channel::Sender;
use rand::distributions::{WeightedIndex, Distribution};
use rand::{thread_rng, Rng};
use std::sync::atomic::{AtomicBool, Ordering};
use itertools::izip;
use crossbeam::{scope, channel};
use ta::indicators::ExponentialMovingAverage;
use ta::Next;
use ordered_float::OrderedFloat;
use closure::closure;
use crate::mcts_zero::{Response, Tree, ZeroState, RunResult, KeepResult, Request, zero_build_tree};
use crate::network::Network;
use crate::network::google_torch::GoogleTorchNetwork;


#[derive(Debug, Clone)]
pub struct Settings {
    // performance settings
    pub devices: Vec<Device>,
    pub threads_per_device: usize,
    pub batch_size: usize,

    // how many positions to generate
    pub position_count: usize,
    pub output_path: String,

    // settings that effect the generated games
    pub network_path: String,
    pub iterations: u64,
    pub exploration_weight: f32,
    pub inf_temp_move_count: u32,
}

// A message sent back from a worker thread to the main collector thread.
#[derive(Debug)]
enum Message {
    Simulation(Simulation),
    Counter { evals: usize, moves: usize },
}

// A full game.
#[derive(Debug, Clone)]
struct Simulation {
    won_by: Player,
    positions: Vec<Position>,
}

// A single position in a game.
#[derive(Debug, Clone)]
struct Position {
    board: Board,
    value: f32,
    policy: Vec<f32>,
}

// The state kept while generating a new game.
#[derive(Debug, Clone)]
struct GameState {
    zero: ZeroState,
    positions: Vec<Position>,
}

type Net = GoogleTorchNetwork;

impl Settings {
    pub fn all_cuda_devices() -> Vec<Device> {
        (0..Cuda::device_count() as usize).map(Device::Cuda).collect_vec()
    }

    pub fn run(&self) {
        println!("{:#?}", self);

        // build the starting board tree once so we don't need to compute it for every started game
        println!("Building root tree");
        let root_tree = self.build_root_tree();

        println!("Starting threads");
        let start = Instant::now();
        let mut last_print = start;

        let (sender, receiver) = channel::unbounded();
        let request_stop = AtomicBool::new(false);

        scope(|s| {
            //spawn threads
            for &device in &self.devices {
                for i in 0..self.threads_per_device {
                    s.builder()
                        .name(format!("worker-{}", i))
                        .spawn(closure!(
                            ref root_tree, ref request_stop, ref sender,
                            |_| thread_main(self, device, root_tree, request_stop, sender)
                        )).unwrap();
                }
            }

            //performance metrics
            let mut total_eval_count = 0;
            let mut total_move_count = 0;
            let mut total_game_count = 0;
            let mut total_pos_count = 0;
            let mut eval_throughput = ExponentialMovingAverage::new(5).unwrap();
            let mut move_throughput = ExponentialMovingAverage::new(5).unwrap();
            let mut game_throughput = ExponentialMovingAverage::new(5).unwrap();
            let mut pos_throughput = ExponentialMovingAverage::new(5).unwrap();
            let mut eval_throughput_cached = f64::NAN;
            let mut move_throughput_cached = f64::NAN;

            //collect results
            for message in &receiver {
                let now = Instant::now;
                let elapsed = (now() - start).as_secs_f64();

                //print metrics every second
                if (now() - last_print).as_secs_f64() > 1.0 {
                    last_print = now();
                    println!("Evals:     {:.2} evals/s => {}", eval_throughput_cached, total_eval_count);
                    println!("Moves:     {:.2} moves/s => {}", move_throughput_cached, total_move_count);
                    println!("Games:     {:.2} games/s => {}", game_throughput.next((total_game_count as f64) / elapsed), total_game_count);
                    println!("Positions: {:.2} pos/s => {}", pos_throughput.next((total_pos_count as f64) / elapsed), total_pos_count);
                    println!();
                }

                //handle incoming message
                match message {
                    Message::Simulation(simulation) => {
                        total_pos_count += simulation.positions.len();
                        total_game_count += 1;

                        let mut values = vec![];
                        let mut factor = 1.0;

                        for p in &simulation.positions {
                            values.push(factor * p.value);
                            factor *= -1.0;
                        }
                    }
                    Message::Counter { moves, evals } => {
                        total_eval_count += evals;
                        total_move_count += moves;
                        eval_throughput_cached = eval_throughput.next((total_eval_count as f64) / elapsed);
                        move_throughput_cached = move_throughput.next((total_move_count as f64) / elapsed);
                    }
                }

                //we have enough positions, stop
                if total_move_count >= self.position_count {
                    break;
                }
            }

            //stop threads, don't drop receiver yet
            request_stop.store(true, Ordering::SeqCst);

            //scope automatically joins spawned threads
        }).unwrap();
    }

    fn load_network(&self, device: Device) -> Net {
        Net::load(&self.network_path, device)
    }

    fn new_zero(&self, tree: Tree) -> ZeroState {
        ZeroState::new(tree, self.iterations, self.exploration_weight)
    }

    fn build_root_tree(&self) -> Tree {
        let mut network = self.load_network(Device::Cpu);
        zero_build_tree(&Board::new(), self.iterations, self.exploration_weight, &mut network)
    }
}

/// The entry point for worker threads.
fn thread_main(
    settings: &Settings,
    device: Device,
    root_tree: &Tree,
    request_stop: &AtomicBool,
    sender: &Sender<Message>
) {
    let batch_size = settings.batch_size;
    let mut network = settings.load_network(device);
    let mut rng = thread_rng();

    let mut states = vec![GameState::new(settings.new_zero(root_tree.clone())); batch_size];
    let mut responses = (0..batch_size).map(|_| None).collect_vec();
    let mut requests = Vec::new();

    while !request_stop.load(Ordering::SeqCst) {
        let mut total_move_count = 0;

        //advance all games and collect the next batch of requests
        for i in 0..batch_size {
            let state = &mut states[i];

            let response = responses[i].take();
            let (move_count, request) = state.run_until_request(&mut rng, settings, root_tree, response, sender);
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

        //count the number of evaluations that happened
        sender.send(Message::Counter { evals: batch_size, moves: total_move_count }).unwrap();
    }
}

impl GameState {
    fn new(zero: ZeroState) -> Self {
        GameState { zero, positions: vec![] }
    }

    fn run_until_request(
        &mut self,
        rng: &mut impl Rng,
        settings: &Settings,
        root_tree: &Tree,
        response: Option<Response>,
        sender: &Sender<Message>,
    ) -> (usize, Request) {
        let mut response = response;
        let mut move_count = 0;

        loop {
            let result = self.zero.run_until_result(response.take());

            match result {
                RunResult::Request(request) =>
                    return (move_count, request),
                RunResult::Done => {
                    let tree = &self.zero.tree;
                    let policy = tree.policy().collect_vec();

                    //pick a move to play
                    let picked_index = if tree.root_board().count_tiles() > settings.inf_temp_move_count {
                        //pick the best move
                        policy.iter().copied().map(OrderedFloat).position_max().unwrap()
                    } else {
                        //pick a random move following the policy
                        WeightedIndex::new(&policy).unwrap().sample(rng)
                    };
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
                            sender.send(Message::Simulation(simulation)).unwrap();

                            //start a new game
                            *self = GameState::new(settings.new_zero(root_tree.clone()));
                        }
                    }
                }
            }
        }
    }
}
