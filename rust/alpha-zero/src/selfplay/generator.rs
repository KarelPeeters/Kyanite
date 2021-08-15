use crossbeam::channel::{Receiver, Sender, SendError, TryRecvError};
use itertools::{Itertools, zip_eq};
use lru::LruCache;
use rand::{Rng, thread_rng};
use rand_distr::Dirichlet;

use board_game::board::{Board, Outcome};
use cuda_sys::wrapper::handle::Device;

use crate::network::Network;
use crate::selfplay::core::{MoveSelector, Position, Simulation};
use crate::selfplay::protocol::{Command, GeneratorUpdate, Settings};
use crate::zero::{KeepResult, Request, Response, RunResult, Tree, ZeroEvaluation, ZeroSettings, ZeroState};

pub fn generator_main<B: Board, N: Network<B>>(
    start_pos: impl Fn() -> B,
    load_network: impl Fn(String, usize, Device) -> N,
    device: Device,
    batch_size: usize,
    cmd_receiver: Receiver<Command>,
    update_sender: Sender<GeneratorUpdate<B>>,
) -> Result<(), SendError<GeneratorUpdate<B>>> {
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
            Ok(Command::StartupSettings(_)) => panic!("Already received startup settings"),
            Ok(Command::NewSettings(new_settings)) => {
                settings = Some(new_settings)
            }
            Ok(Command::NewNetwork(path)) => {
                println!("Generator thread loading new network {:?}", path);
                network = Some(load_network(path, batch_size, device));
            }
        }

        // advance generator
        if let Some(settings) = &settings {
            if let Some(executor) = &mut network {
                state.cache.resize(settings.cache_size);
                state.step(&start_pos, &update_sender, settings, executor, batch_size, &mut rng)?;
            }
        }
    }

    Ok(())
}

#[derive(Debug)]
struct GeneratorState<B: Board> {
    games: Vec<GameState<B>>,
    responses: Vec<Response<B>>,
    cache: LruCache<B, ZeroEvaluation>,
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
        GeneratorState { games: Default::default(), responses: Default::default(), cache: LruCache::new(0) }
    }

    fn step(
        &mut self,
        start_pos: impl Fn() -> B,
        update_sender: &Sender<GeneratorUpdate<B>>,
        settings: &Settings,
        network: &mut impl Network<B>,
        batch_size: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SendError<GeneratorUpdate<B>>> {
        let mut requests: Vec<Request<B>> = vec![];

        let mut total_cached_eval_count = 0;
        let mut total_move_count = 0;

        // run all existing games and collect the requests
        assert_eq!(self.games.len(), self.responses.len());
        assert!(requests.is_empty());
        let mut kept_games = vec![];

        for (mut game, response) in zip_eq(self.games.drain(..), self.responses.drain(..)) {
            let (request, cached_eval_count, move_count) =
                game.run_until_request(rng, &mut self.cache, settings, Some(response), &update_sender);

            total_cached_eval_count += cached_eval_count;
            total_move_count += move_count;

            if let Some(request) = request {
                // this game is not done, keep it and its request
                kept_games.push(game);
                requests.push(request);
            }
        }

        self.games = kept_games;

        // create new games until we have enough and run them once
        let new_game_count = batch_size.saturating_sub(self.games.len());
        for _ in 0..new_game_count {
            let zero = new_zero(settings, Tree::new(start_pos()), rng);
            let mut game = GameState::new(zero);
            let (request, cached_eval_count, move_count) =
                game.run_until_request(rng, &mut self.cache, settings, None, &update_sender);

            total_cached_eval_count += cached_eval_count;
            total_move_count += move_count;

            let request = request.expect("The first run of a gamestate should always returns a request");

            self.games.push(game);
            requests.push(request);
        }

        let request_count = requests.len();

        //pass requests to network
        assert!(self.responses.is_empty());
        self.responses = network.evaluate_batch_requests(&requests);

        //insert responses into the cache
        for response in &self.responses {
            self.cache.put(response.request.board(), response.evaluation.clone());
        }

        //send the number of evaluations that happened
        update_sender.send(GeneratorUpdate::Progress {
            real_evals: request_count as u64,
            cached_evals: total_cached_eval_count,
            moves: total_move_count,
        })?;

        Ok(())
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
        cache: &mut LruCache<B, ZeroEvaluation>,
        settings: &Settings,
        response: Option<Response<B>>,
        sender: &Sender<GeneratorUpdate<B>>,
    ) -> (Option<Request<B>>, u64, u64) {
        let move_selector = MoveSelector::new(settings.temperature, settings.zero_temp_move_count);

        let mut response = response;
        let mut move_count = 0;
        let mut cached_eval_count = 0;

        let request = loop {
            let had_response = response.is_some();
            let result = self.zero.run_until_result(response.take(), rng);

            if had_response && self.needs_dirichlet {
                // at this point we're sure that the tree has at least one visit, se we can add the noise
                add_dirichlet_noise(settings, &mut self.zero.tree, rng);
                self.needs_dirichlet = false;
            }

            match result {
                RunResult::Request(request) => {
                    if let Some(evaluation) = cache.get(&request.board()) {
                        response = Some(Response { request, evaluation: evaluation.clone() });
                        cached_eval_count += 1;
                    } else {
                        break Some(request);
                    }
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
                            break None;
                        }
                    }
                }
            }
        };

        (request, cached_eval_count, move_count)
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
