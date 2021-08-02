use crossbeam::channel::{Receiver, Sender, TryRecvError};
use itertools::{Itertools, zip_eq};
use rand::{Rng, thread_rng};
use rand_distr::Dirichlet;

use board_game::board::{Board, Outcome};

use crate::network::Network;
use crate::new_selfplay::core::{Command, Settings, GeneratorUpdate};
use crate::selfplay::{MoveSelector, Position, Simulation};
use crate::zero::{KeepResult, Request, Response, RunResult, Tree, ZeroEvaluation, ZeroSettings, ZeroState};
use cuda_sys::wrapper::handle::Device;

pub fn generator_main<B: Board, N: Network<B>>(
    start_pos: impl Fn() -> B,
    load_network: impl Fn(String, Device) -> N,
    device: Device,
    batch_size: usize,
    cmd_receiver: Receiver<Command>,
    update_sender: Sender<GeneratorUpdate<B>>,
) {
    let mut state = GeneratorState::new();
    let mut rng = thread_rng();

    let mut settings = None;
    let mut network = None;

    loop {
        // If we don't yet have settings and an executor, block until we get a message.
        // Otherwise only check for new messages without blocking.
        let cmd = if settings.is_some() && network.is_some() {
            cmd_receiver.try_recv()
        } else {
            cmd_receiver.recv()
                .map_err(|_| TryRecvError::Disconnected)
        };

        match cmd {
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => panic!("Channel disconnected"),
            Ok(Command::Stop) => break,
            Ok(Command::NewSettings(new_settings)) => {
                settings = Some(new_settings)
            }
            Ok(Command::NewNetwork(path)) => {
                network = Some(load_network(path, device));
            }
        }

        // advance generator
        if let Some(settings) = &settings {
            if let Some(executor) = &mut network {
                state.step(&start_pos, &update_sender, settings, executor, batch_size, &mut rng)
            }
        }
    }
}

#[derive(Debug)]
struct GeneratorState<B: Board> {
    games: Vec<GameState<B>>,
}

/// The state kept while generating a new game.
#[derive(Debug)]
struct GameState<B: Board> {
    zero: ZeroState<B>,
    needs_dirichlet: bool,
    positions: Vec<Position<B>>,
}

impl<B: Board> GeneratorState<B> {
    fn new() -> Self {
        GeneratorState { games: vec![] }
    }

    fn step(&mut self,
            start_pos: impl Fn() -> B,
            update_sender: &Sender<GeneratorUpdate<B>>,
            settings: &Settings,
            network: &mut impl Network<B>,
            batch_size: usize,
            rng: &mut impl Rng,
    ) {
        let move_selector = MoveSelector::new(settings.temperature, settings.zero_temp_move_count);

        let mut games: Vec<GameState<B>> = vec![];
        let mut requests: Vec<Request<B>> = vec![];
        let mut responses: Vec<Response<B>> = vec![];

        loop {
            let mut total_move_count = 0;

            // run all existing games and collect the requests
            assert_eq!(games.len(), responses.len());
            assert!(requests.is_empty());
            let mut kept_games = vec![];

            for (mut game, response) in zip_eq(games.drain(..), responses.drain(..)) {
                let (request, move_count) = game.run_until_request(rng, move_selector, settings, Some(response), &update_sender);
                total_move_count += move_count;

                if let Some(request) = request {
                    // this game is not done, keep it and its request
                    kept_games.push(game);
                    requests.push(request);
                }
            }

            games = kept_games;

            // create new games until we have enough and run them once
            let new_game_count = batch_size.saturating_sub(games.len());
            for _ in 0..new_game_count {
                let zero = new_zero(settings, Tree::new(start_pos()), rng);
                let mut game = GameState::new(zero);
                let (request, move_count) = game.run_until_request(rng, move_selector, settings, None, &update_sender);

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
            update_sender.send(GeneratorUpdate::Progress {
                real_evals: request_count as u64,
                cached_evals: 0,
                moves: total_move_count,
            }).unwrap();
        }
    }
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
        settings: &Settings,
        response: Option<Response<B>>,
        sender: &Sender<GeneratorUpdate<B>>,
    ) -> (Option<Request<B>>, u64) {
        let mut response = response;
        let mut move_count = 0;

        loop {
            let had_response = response.is_some();
            let result = self.zero.run_until_result(response.take(), rng);

            if had_response && self.needs_dirichlet {
                // at this point we're sure that the tree has at least one visit, se we can add the noise
                add_dirichlet_noise(settings, &mut self.zero.tree, rng);
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
                    let result = if self.positions.len() as u64 >= settings.max_game_length {
                        KeepResult::Done(Outcome::Draw)
                    } else {
                        tree.keep_move(picked_move)
                    };

                    match result {
                        KeepResult::Tree(next_tree) => {
                            //continue playing this game, either by keeping part of the tree or starting a new one on the next board
                            if settings.keep_tree {
                                self.zero = new_zero(settings, next_tree, rng)
                            } else {
                                self.zero = new_zero(settings, Tree::new(next_tree.root_board().clone()), rng)
                            }
                        }
                        KeepResult::Done(outcome) => {
                            //record this game
                            let simulation = Simulation { outcome, positions: std::mem::take(&mut self.positions) };
                            sender.send(GeneratorUpdate::FinishedSimulation(simulation)).unwrap();

                            //report that this game is done
                            return (None, move_count);
                        }
                    }
                }
            }
        }
    }
}

fn add_dirichlet_noise<B: Board>(settings: &Settings, tree: &mut Tree<B>, rng: &mut impl Rng) {
    let children = tree[0].children
        .expect("root node has no children yet, it must have been visited at least once");

    if children.length > 1 {
        let distr = Dirichlet::new_with_size(settings.dirichlet_alpha, children.length as usize).unwrap();
        let noise = rng.sample(distr);

        for (child, n) in zip_eq(children, noise) {
            tree[child].net_policy += n
        }
    }
}

fn new_zero<B: Board>(settings: &Settings, tree: Tree<B>, rng: &mut impl Rng) -> ZeroState<B> {
    let iterations = if rng.gen_bool(settings.full_search_prob) {
        settings.full_iterations
    } else {
        settings.part_iterations
    };

    let zero_settings = ZeroSettings::new(settings.exploration_weight, settings.random_symmetries);
    ZeroState::new(tree, iterations, zero_settings)
}
