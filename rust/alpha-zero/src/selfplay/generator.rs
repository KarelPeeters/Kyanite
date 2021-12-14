use board_game::board::Board;
use board_game::games::max_length::MaxMovesBoard;
use crossbeam::channel::{Receiver, SendError, TryRecvError};
use itertools::Itertools;
use rand::Rng;
use rand::rngs::ThreadRng;
use rand_distr::Dirichlet;

use cuda_nn_eval::Device;
use nn_graph::onnx::load_graph_from_onnx_path;
use nn_graph::optimizer::{optimize_graph, OptimizerSettings};

use crate::mapping::BoardMapper;
use crate::network::{Network, ZeroEvaluation};
use crate::network::cudnn::CudnnNetwork;
use crate::network::symmetry::RandomSymmetryNetwork;
use crate::oracle::DummyOracle;
use crate::selfplay::move_selector::MoveSelector;
use crate::selfplay::protocol::{Command, GeneratorUpdate, Settings};
use crate::selfplay::simulation::{Position, Simulation};
use crate::util::zip_eq_exact;
use crate::zero::step::{FpuMode, zero_step_apply, zero_step_gather, ZeroRequest, ZeroResponse};
use crate::zero::tree::Tree;

pub fn generator_main<B: Board>(
    thread_id: usize,
    mapper: impl BoardMapper<B>,
    start_pos: impl Fn() -> B,
    device: Device,
    batch_size: usize,
    cmd_receiver: Receiver<Command>,
    sender: UpdateSender<B>,
) -> Result<(), SendError<GeneratorUpdate<B>>> {
    let mut state = GeneratorState::new(batch_size);

    //TODO try with a different(faster) rng
    let mut rng = RngType::default();

    let mut settings = None;
    let mut network = None;
    let mut next_index = 0;

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
            Ok(Command::StartupSettings(_)) => panic!("Already received startup settings"),
            Ok(Command::Stop) => break,
            Ok(Command::WaitForNewNetwork) => {
                network = None;
            }
            Ok(Command::NewSettings(new_settings)) => {
                settings = Some(new_settings)
            }
            Ok(Command::NewNetwork(path)) => {
                println!("Generator thread loading new network {:?}", path);
                let loaded_graph = load_graph_from_onnx_path(path);
                let graph = optimize_graph(&loaded_graph, OptimizerSettings::default());
                network = Some(CudnnNetwork::new(mapper, graph, batch_size, device));
            }
        }

        // advance generator
        if let Some(settings) = &settings {
            if let Some(network) = &mut network {
                let mut ctx = Context { thread_id, next_index, settings: &settings, rng: &mut rng };
                state.step(&mut ctx, &start_pos, network, &sender)?;
                next_index = ctx.next_index;
            }
        }
    }

    Ok(())
}

type UpdateSender<B> = crossbeam::channel::Sender<GeneratorUpdate<B>>;
type RngType = ThreadRng;

#[derive(Debug)]
struct Context<'a> {
    thread_id: usize,
    next_index: u64,

    settings: &'a Settings,
    rng: &'a mut RngType,
}

#[derive(Debug)]
struct GeneratorState<B: Board> {
    games: Vec<GameState<B>>,
    responses: Vec<ZeroResponse>,
    batch_size: usize,
}

#[derive(Debug)]
struct GameState<B: Board> {
    index: u64,
    search: SearchState<B>,
    positions: Vec<Position<B>>,
}

#[derive(Debug)]
struct SearchState<B: Board> {
    tree: Tree<MaxMovesBoard<B>>,
    needs_dirichlet: bool,
    is_full_search: bool,
    root_net_eval: Option<ZeroEvaluation>,
}

#[derive(Debug)]
enum StepResult<B: Board> {
    Done,
    Request(ZeroRequest<MaxMovesBoard<B>>),
}

#[derive(Debug, Default)]
struct Counter {
    move_count: u64,
    cache_hits: u64,
}

impl<B: Board> GeneratorState<B> {
    fn new(batch_size: usize) -> Self {
        GeneratorState {
            games: vec![],
            responses: vec![],
            batch_size,
        }
    }

    fn step(
        &mut self,
        ctx: &mut Context,
        start_pos: impl Fn() -> B,
        network: impl Network<B>,
        sender: &UpdateSender<B>,
    ) -> Result<(), SendError<GeneratorUpdate<B>>> {
        let mut counter = Counter::default();
        let requests = self.collect_requests(ctx, &mut counter, sender, start_pos);
        assert_eq!(requests.len(), self.batch_size);

        // evaluate the requests
        //TODO kind of sketchy that the network doesn't get to see the move counter, is that okay?
        let boards = requests.iter().map(|r| r.board.inner()).collect_vec();
        let evals = RandomSymmetryNetwork::new(network, &mut ctx.rng, ctx.settings.random_symmetries)
            .evaluate_batch(&boards);

        // store the responses for next step
        assert!(self.responses.is_empty());
        self.responses.extend(
            zip_eq_exact(requests, evals)
                .map(|(req, eval)| req.respond(eval))
        );

        // report progress
        sender.send(GeneratorUpdate::Progress {
            cached_evals: counter.cache_hits,
            real_evals: self.batch_size as u64,
            moves: counter.move_count,
        })?;

        Ok(())
    }

    fn collect_requests(
        &mut self,
        ctx: &mut Context,
        counter: &mut Counter,
        sender: &UpdateSender<B>,
        start_pos: impl Fn() -> B,
    ) -> Vec<ZeroRequest<MaxMovesBoard<B>>> {
        let mut requests = vec![];
        let existing_games = std::mem::take(&mut self.games);

        let mut step_and_append = |ctx: &mut Context, games: &mut Vec<GameState<B>>, mut game: GameState<B>, response: Option<ZeroResponse>| {
            let result = game.step(ctx, response, sender, counter);

            match result {
                StepResult::Done => {}
                StepResult::Request(request) => {
                    games.push(game);
                    requests.push(request);
                }
            }
        };

        // step all existing games
        for (game, response) in zip_eq_exact(existing_games, self.responses.drain(..)) {
            step_and_append(ctx, &mut self.games, game, Some(response))
        }

        // start new games until we have enough of them
        while self.games.len() < self.batch_size {
            let game = GameState::new(ctx, start_pos());
            step_and_append(ctx, &mut self.games, game, None);
        }

        assert_eq!(requests.len(), self.games.len());
        requests
    }
}

impl<B: Board> GameState<B> {
    fn new(ctx: &mut Context, start_pos: B) -> Self {
        let tree = Tree::new(MaxMovesBoard::new(start_pos, ctx.max_moves()));
        let index = ctx.next_index;
        ctx.next_index += 1;
        GameState {
            index,
            search: SearchState::new(ctx, tree),
            positions: vec![],
        }
    }

    fn step(
        &mut self,
        ctx: &mut Context,
        initial_response: Option<ZeroResponse>,
        sender: &UpdateSender<B>,
        counter: &mut Counter,
    ) -> StepResult<B> {
        let mut response = initial_response;

        loop {
            let result = self.search.step(ctx, response.take());

            match result {
                StepResult::Request(request) => {
                    return StepResult::Request(request);
                }
                StepResult::Done => {
                    counter.move_count += 1;
                    if self.search_done_step(ctx, sender) {
                        return StepResult::Done;
                    }
                }
            }
        }
    }

    fn search_done_step(&mut self, ctx: &mut Context, sender: &UpdateSender<B>) -> bool {
        let settings = ctx.settings;

        let tree = &self.search.tree;
        let policy = tree.policy().collect_vec();

        //pick a move to play
        let move_selector = MoveSelector::new(settings.temperature, settings.zero_temp_move_count);
        let picked_index = move_selector.select(self.positions.len() as u32, policy.iter().copied(), ctx.rng);
        let picked_child = tree[0].children.unwrap().get(picked_index);
        let picked_move = tree[picked_child].last_move.unwrap();

        //store this position
        let net_evaluation = self.search.root_net_eval.take().unwrap();
        let zero_evaluation = ZeroEvaluation { values: tree.values(), policy };

        self.positions.push(Position {
            board: tree.root_board().inner().clone(),
            should_store: self.search.is_full_search,
            zero_visits: tree.root_visits(),
            net_evaluation,
            zero_evaluation,
        });

        let mut next_board = tree.root_board().clone();
        next_board.play(picked_move);

        if let Some(outcome) = next_board.outcome() {
            //record this game
            let simulation = Simulation { outcome, positions: std::mem::take(&mut self.positions) };
            sender.send(GeneratorUpdate::FinishedSimulation {
                thread_id: ctx.thread_id,
                index: self.index,
                simulation,
            }).unwrap();

            //report that this game is done
            true
        } else {
            //continue playing this game, either by keeping part of the tree or starting a new one on the next board
            let next_tree = if settings.keep_tree {
                // we already know the next board is not done
                tree.keep_move(picked_move).unwrap()
            } else {
                Tree::new(next_board)
            };
            self.search = SearchState::new(ctx, next_tree);

            // report that this game is not done
            false
        }
    }
}

impl<B: Board> SearchState<B> {
    fn new(ctx: &mut Context, tree: Tree<MaxMovesBoard<B>>) -> Self {
        SearchState {
            tree,
            needs_dirichlet: true,
            is_full_search: ctx.rng.gen_bool(ctx.settings.full_search_prob),
            root_net_eval: None,
        }
    }

    fn step(&mut self, ctx: &mut Context, response: Option<ZeroResponse>) -> StepResult<B> {
        let settings = ctx.settings;

        if let Some(response) = response {
            zero_step_apply(&mut self.tree, response);
        }

        loop {
            if self.tree.root_visits() > 0 && self.needs_dirichlet {
                self.root_net_eval = Some(extract_root_net_eval(&self.tree));
                add_dirichlet_noise(ctx, &mut self.tree);
                self.needs_dirichlet = false;
            }

            let target_iterations = if self.is_full_search { settings.full_iterations } else { settings.part_iterations };
            if self.tree.root_visits() >= target_iterations {
                return StepResult::Done;
            }

            //TODO use an oracle here (based on a boolean or maybe path setting)
            if let Some(request) = zero_step_gather(&mut self.tree, &DummyOracle, settings.exploration_weight, settings.use_value, FpuMode::Parent) {
                return StepResult::Request(request);
            }
        }
    }
}

fn extract_root_net_eval<B: Board>(tree: &Tree<B>) -> ZeroEvaluation {
    let values = tree[0].net_values.unwrap();
    let policy = tree[0].children.unwrap().iter()
        .map(|c| tree[c].net_policy)
        .collect();
    ZeroEvaluation { values, policy }
}

fn add_dirichlet_noise<B: Board>(ctx: &mut Context, tree: &mut Tree<B>) {
    let alpha = ctx.settings.dirichlet_alpha;
    let eps = ctx.settings.dirichlet_eps;

    let children = tree[0].children
        .expect("root node has no children yet, it must have been visited at least once");

    if children.length > 1 {
        let distr = Dirichlet::new_with_size(alpha, children.length as usize).unwrap();
        let noise = ctx.rng.sample(distr);

        for (child, n) in zip_eq_exact(children, noise) {
            let p = &mut tree[child].net_policy;
            *p = (1.0 - eps) * (*p) + eps * n;
        }
    }
}

impl<'a> Context<'a> {
    fn max_moves(&self) -> u64 {
        if self.settings.max_game_length > 0 {
            self.settings.max_game_length as u64
        } else {
            u64::MAX
        }
    }
}