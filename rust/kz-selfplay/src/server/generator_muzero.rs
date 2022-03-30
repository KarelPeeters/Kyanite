use std::borrow::Cow;

use board_game::board::Board;
use crossbeam::channel::{Receiver, SendError, TryRecvError};
use internal_iterator::InternalIterator;
use itertools::Itertools;
use rand::Rng;
use rand::rngs::ThreadRng;
use rand_distr::Dirichlet;

use cuda_nn_eval::Device;
use kz_core::mapping::BoardMapper;
use kz_core::muzero::MuZeroEvaluation;
use kz_core::muzero::step::{
    muzero_step_apply, muzero_step_gather, MuZeroExpandRequest, MuZeroRequest, MuZeroResponse,
};
use kz_core::muzero::tree::MuTree;
use kz_core::network::common::normalize_in_place;
use kz_core::network::muzero::{MuZeroExpandExecutor, MuZeroGraphs, MuZeroRootExecutor};
use kz_core::network::ZeroEvaluation;
use kz_core::zero::step::FpuMode;
use kz_util::zip_eq_exact;
use nn_graph::optimizer::OptimizerSettings;

use crate::move_selector::MoveSelector;
use crate::server::protocol::{Command, GeneratorUpdate, Settings};
use crate::simulation::{Position, Simulation};

//TODO support max moves somehow?
pub fn generator_muzero_main<B: Board>(
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
    let mut executors = None;
    let mut next_index = 0;

    loop {
        // If we don't yet have settings and an executor, block until we get a message.
        // Otherwise only check for new messages without blocking.
        let cmd = if settings.is_some() && executors.is_some() {
            cmd_receiver.try_recv()
        } else {
            cmd_receiver.recv().map_err(|_| TryRecvError::Disconnected)
        };

        match cmd {
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => panic!("Channel disconnected"),
            Ok(Command::StartupSettings(_)) => panic!("Already received startup settings"),
            Ok(Command::Stop) => break,
            Ok(Command::WaitForNewNetwork) => {
                executors = None;
            }
            Ok(Command::NewSettings(new_settings)) => settings = Some(new_settings),
            Ok(Command::NewNetwork(path)) => {
                println!("Generator thread loading new network {:?}", path);

                let graphs = MuZeroGraphs::load(&path, mapper);
                let graphs = graphs.optimize(OptimizerSettings::default());
                let fused = graphs.fuse(OptimizerSettings::default());

                //TODO consider larger batch size for root (maybe 4?) when moving it to a separate thread
                let root_exec = fused.root_executor(device, 1);
                let expand_exec = fused.expand_executor(device, batch_size);

                executors = Some((root_exec, expand_exec));
            }
        }

        // advance generator
        if let Some(settings) = &settings {
            if let Some((root_exec, expand_exec)) = &mut executors {
                let mut ctx = Context {
                    thread_id,
                    next_index,
                    settings,
                    rng: &mut rng,
                    mapper,
                };
                state.step(&mut ctx, &start_pos, root_exec, expand_exec, &sender)?;
                next_index = ctx.next_index;
            }
        }
    }

    Ok(())
}

type UpdateSender<B> = crossbeam::channel::Sender<GeneratorUpdate<B>>;
type RngType = ThreadRng;

#[derive(Debug)]
struct Context<'a, M> {
    thread_id: usize,
    next_index: u64,

    settings: &'a Settings,
    rng: &'a mut RngType,

    mapper: M,
}

#[derive(Debug)]
struct GeneratorState<B: Board> {
    games: Vec<GameState<B>>,
    responses: Vec<MuZeroResponse<'static>>,
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
    tree: MuTree<B>,
    is_full_search: bool,
    root_net_eval: Option<ZeroEvaluation<'static>>,
}

#[derive(Debug)]
enum StepResult<R> {
    Done,
    Request(R),
}

#[derive(Debug, Default)]
struct Counter {
    move_count: u64,
}

impl<B: Board> GeneratorState<B> {
    fn new(batch_size: usize) -> Self {
        GeneratorState {
            games: vec![],
            responses: vec![],
            batch_size,
        }
    }

    fn step<M: BoardMapper<B>>(
        &mut self,
        ctx: &mut Context<M>,
        start_pos: impl Fn() -> B,
        root_exec: &mut MuZeroRootExecutor<B, M>,
        expand_exec: &mut MuZeroExpandExecutor<B, M>,
        sender: &UpdateSender<B>,
    ) -> Result<(), SendError<GeneratorUpdate<B>>> {
        let mut counter = Counter::default();
        let requests = self.collect_requests(ctx, &mut counter, root_exec, sender, start_pos);
        assert_eq!(requests.len(), self.batch_size);

        // evaluate the requests
        let pairs = requests.iter().map(|r| (r.state.clone(), r.move_index)).collect_vec();
        let evals = expand_exec.eval_expand(&pairs);

        // store the responses for next step
        assert!(self.responses.is_empty());
        self.responses.extend(
            zip_eq_exact(requests, evals).map(|(req, (state, eval))| MuZeroResponse {
                node: req.node,
                state,
                eval,
            }),
        );

        // report progress
        sender.send(GeneratorUpdate::Progress {
            cached_evals: 0,
            real_evals: self.batch_size as u64,
            moves: counter.move_count,
        })?;

        Ok(())
    }

    fn collect_requests<M: BoardMapper<B>>(
        &mut self,
        ctx: &mut Context<M>,
        counter: &mut Counter,
        root_exec: &mut MuZeroRootExecutor<B, M>,
        sender: &UpdateSender<B>,
        start_pos: impl Fn() -> B,
    ) -> Vec<MuZeroExpandRequest> {
        let mut requests = vec![];
        let existing_games = std::mem::take(&mut self.games);

        let mut step_and_append = |ctx: &mut Context<M>,
                                   games: &mut Vec<GameState<B>>,
                                   mut game: GameState<B>,
                                   response: Option<MuZeroResponse>| {
            let result = game.step(ctx, response, root_exec, sender, counter);

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
    fn new<M: BoardMapper<B>>(ctx: &mut Context<M>, start_pos: B) -> Self {
        let tree = MuTree::new(start_pos, ctx.mapper.policy_len());
        let index = ctx.next_index;
        ctx.next_index += 1;
        GameState {
            index,
            search: SearchState::new(ctx, tree),
            positions: vec![],
        }
    }

    fn step<M: BoardMapper<B>>(
        &mut self,
        ctx: &mut Context<M>,
        initial_response: Option<MuZeroResponse>,
        root_exec: &mut MuZeroRootExecutor<B, M>,
        sender: &UpdateSender<B>,
        counter: &mut Counter,
    ) -> StepResult<MuZeroExpandRequest> {
        let mut response = initial_response;

        loop {
            let result = self.search.step(ctx, response.take());

            match result {
                StepResult::Request(request) => {
                    match request {
                        MuZeroRequest::Root { node, board } => {
                            let mut result = root_exec.eval_root(&[board]);
                            assert_eq!(result.len(), 1);
                            let (state, eval) = result.remove(0);

                            response = Some(MuZeroResponse { node, state, eval })
                            // continue the loop with the new root response
                        }
                        MuZeroRequest::Expand(req) => {
                            return StepResult::Request(req);
                        }
                    }
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

    fn search_done_step<M: BoardMapper<B>>(&mut self, ctx: &mut Context<M>, sender: &UpdateSender<B>) -> bool {
        let settings = ctx.settings;

        let tree = &self.search.tree;

        // extract both evaluations
        let net_evaluation = self.search.root_net_eval.take().unwrap();
        let zero_evaluation = tree.eval();

        //pick a move to play
        let move_selector = MoveSelector::new(settings.temperature, settings.zero_temp_move_count);
        let picked_index = move_selector.select(self.positions.len() as u32, zero_evaluation.policy.as_ref(), ctx.rng);
        let picked_child = tree[0].inner.as_ref().unwrap().children.get(picked_index);
        let picked_move_index = tree[picked_child].last_move_index.unwrap();
        let picked_move = ctx.mapper.index_to_move(tree.root_board(), picked_move_index).unwrap();

        // store this position
        self.positions.push(Position {
            board: tree.root_board().clone(),
            should_store: self.search.is_full_search,
            played_mv: picked_move,
            zero_visits: tree.root_visits(),
            net_evaluation,
            zero_evaluation,
        });

        let mut next_board = tree.root_board().clone();
        next_board.play(picked_move);

        if let Some(outcome) = next_board.outcome() {
            // record this game
            let simulation = Simulation {
                outcome,
                positions: std::mem::take(&mut self.positions),
            };
            sender
                .send(GeneratorUpdate::FinishedSimulation {
                    thread_id: ctx.thread_id,
                    index: self.index,
                    simulation,
                })
                .unwrap();

            // report that this game is done
            true
        } else {
            // continue playing this game, either by keeping part of the tree or starting a new one on the next board
            let next_tree = MuTree::new(next_board, ctx.mapper.policy_len());
            self.search = SearchState::new(ctx, next_tree);

            // report that this game is not done
            false
        }
    }
}

impl<B: Board> SearchState<B> {
    fn new<M: BoardMapper<B>>(ctx: &mut Context<M>, tree: MuTree<B>) -> Self {
        SearchState {
            tree,
            is_full_search: ctx.rng.gen_bool(ctx.settings.full_search_prob),
            root_net_eval: None,
        }
    }

    fn step<M: BoardMapper<B>>(
        &mut self,
        ctx: &mut Context<M>,
        response: Option<MuZeroResponse>,
    ) -> StepResult<MuZeroRequest<B>> {
        let settings = ctx.settings;

        if let Some(mut response) = response {
            if response.node == 0 {
                self.root_net_eval = Some(extract_zero_eval(ctx.mapper, self.tree.root_board(), &response.eval));
                add_dirichlet_noise(ctx, self.tree.root_board(), response.eval.policy.to_mut());
            }

            let top_moves = if ctx.settings.top_moves == 0 {
                usize::MAX
            } else {
                ctx.settings.top_moves
            };
            muzero_step_apply(&mut self.tree, top_moves, response, ctx.mapper);
        }

        loop {
            let target_iterations = if self.is_full_search {
                settings.full_iterations
            } else {
                settings.part_iterations
            };
            if self.tree.root_visits() >= target_iterations {
                return StepResult::Done;
            }

            //TODO use an oracle here (based on a boolean or maybe path setting)
            if let Some(request) = muzero_step_gather(
                &mut self.tree,
                settings.weights.to_uct(),
                settings.use_value,
                FpuMode::Parent,
            ) {
                return StepResult::Request(request);
            }
        }
    }
}

fn add_dirichlet_noise<B: Board, M: BoardMapper<B>>(ctx: &mut Context<M>, board: &B, policy: &mut [f32]) {
    let alpha = ctx.settings.dirichlet_alpha;
    let eps = ctx.settings.dirichlet_eps;

    let mv_count = board.available_moves().count();
    if mv_count > 1 {
        let indices = || {
            board
                .available_moves()
                .map(|mv| ctx.mapper.move_to_index(board, mv).unwrap())
        };

        let mut total_p = 0.0;
        indices().for_each(|pi| total_p += policy[pi]);

        let distr = Dirichlet::new_with_size(alpha, mv_count).unwrap();
        let noise = ctx.rng.sample(distr);

        indices().enumerate().for_each(|(i, pi)| {
            policy[pi] = (policy[pi] / total_p) * (1.0 - eps) + noise[i] * eps;
        });
    }
}

fn extract_zero_eval<B: Board, M: BoardMapper<B>>(
    mapper: M,
    board: &B,
    response: &MuZeroEvaluation,
) -> ZeroEvaluation<'static> {
    let mut policy: Vec<f32> = board
        .available_moves()
        .map(|mv| mapper.move_to_index(board, mv).map_or(1.0, |i| response.policy[i]))
        .collect();

    // TODO should we even normalize here? that just means we lose valid weight information
    normalize_in_place(&mut policy);

    ZeroEvaluation {
        values: response.values,
        policy: Cow::Owned(policy),
    }
}
