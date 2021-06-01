use sttt_zero::network::Network;
use tch::{Device, Cuda};
use sttt_zero::mcts_zero::{ZeroState, Tree, RunResult, Response, Request};
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


#[derive(Debug, Clone)]
struct Settings {
    threads_per_device: usize,
    batch_size: usize,

    network_path: String,
    iterations: u64,
    exploration_weight: f32,

    inf_temp_move_count: u32,

    // move_caching: bool,
    // tree_reuse: bool,

    position_count: usize,
}

fn main() {
    let devices = (0..Cuda::device_count() as usize).map(Device::Cuda).collect_vec();
    let settings = Settings {
        threads_per_device: 2,
        batch_size: 200,

        iterations: 1000,
        exploration_weight: 1.0,
        network_path: "../data/esat/trained_model_10_epochs.pt".to_owned(),

        inf_temp_move_count: 20,

        position_count: 100_000,
    };

    println!("Devices {{ {:?} }}", devices);
    println!("{:#?}", settings);

    let start = Instant::now();
    let mut last_print = start;

    let (sender, receiver) = channel::unbounded();
    let request_stop = AtomicBool::new(false);

    scope(|s| {
        //spawn threads
        for &device in &devices {
            for _ in 0..settings.threads_per_device {
                s.spawn(closure!(
                    ref settings, ref request_stop, ref sender,
                    |_| thread_main(settings, device, request_stop, sender)
                ));
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

                    // println!("{:?} => {}", values, simulation.won_by.sign());
                }
                Message::Counter { moves, evals } => {
                    total_eval_count += evals;
                    total_move_count += moves;
                    eval_throughput_cached = eval_throughput.next((total_eval_count as f64) / elapsed);
                    move_throughput_cached = move_throughput.next((total_move_count as f64) / elapsed);
                }
            }

            //we have enough positions, stop
            if total_move_count >= settings.position_count {
                break;
            }
        }

        //stop threads, don't drop receiver yet
        request_stop.store(true, Ordering::SeqCst);

        //scope automatically joins spawned threads
    }).unwrap();
}

enum Message {
    Simulation(Simulation),
    Counter { evals: usize, moves: usize },
}

#[allow(dead_code)]
struct Simulation {
    won_by: Player,
    positions: Vec<Position>,
}

#[allow(dead_code)]
struct Position {
    board: Board,
    value: f32,
    policy: Vec<f32>,
}

impl Settings {
    fn load_network(&self, device: Device) -> Network {
        Network::load(&self.network_path, device)
    }

    fn new_zero(&self, board: Board) -> ZeroState {
        ZeroState::new(Tree::new(board), self.iterations, self.exploration_weight)
    }
}

struct GameState {
    zero: ZeroState,
    positions: Vec<Position>,
}

impl GameState {
    fn new(settings: &Settings) -> Self {
        GameState {
            zero: settings.new_zero(Board::new()),
            positions: Default::default(),
        }
    }

    fn run_until_request(&mut self, rng: &mut impl Rng, settings: &Settings, response: Option<Response>, sender: &Sender<Message>) -> (usize, Request) {
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
                    let picked_child = tree[0].children().unwrap().get(picked_index);
                    let picked_move = tree[picked_child].coord;

                    //store this position
                    self.positions.push(Position {
                        board: self.zero.tree.root_board().clone(),
                        value: tree[0].value(),
                        policy,
                    });

                    //actually play the move
                    let mut next_board = tree.root_board().clone();
                    next_board.play(picked_move);
                    move_count += 1;

                    match next_board.won_by {
                        None => {
                            //continue playing this game
                            self.zero = settings.new_zero(next_board)
                        }
                        Some(won_by) => {
                            //record this game
                            let simulation = Simulation { won_by, positions: std::mem::take(&mut self.positions) };
                            sender.send(Message::Simulation(simulation)).unwrap();

                            //start a new game
                            self.zero = settings.new_zero(Board::new());
                        }
                    }
                }
            }
        }
    }
}

fn thread_main(settings: &Settings, device: Device, request_stop: &AtomicBool, sender: &Sender<Message>) {
    let batch_size = settings.batch_size;
    let mut network = settings.load_network(device);
    let mut rng = thread_rng();

    let mut states = (0..batch_size).map(|_| GameState::new(settings)).collect_vec();
    let mut responses = (0..batch_size).map(|_| None).collect_vec();
    let mut requests = Vec::new();


    while !request_stop.load(Ordering::SeqCst) {
        let mut total_move_count = 0;

        //advance all games and collect the next batch of requests
        for i in 0..batch_size {
            let state = &mut states[i];

            let response = responses[i].take();
            let (move_count, request) = state.run_until_request(&mut rng, settings, response, sender);
            total_move_count += move_count;
            requests.push(request);
        }

        //pass requests to network
        let boards = requests.iter().map(|r| r.board.clone()).collect_vec();
        let mut evaluations = network.evaluate_all(&boards);

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
