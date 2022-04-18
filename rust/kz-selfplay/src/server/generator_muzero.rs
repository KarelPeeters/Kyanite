use std::borrow::Cow;

use board_game::board::Board;
use crossbeam::channel::{Receiver, Select, SendError, TryRecvError};
use internal_iterator::InternalIterator;
use rand::rngs::ThreadRng;
use rand::Rng;
use rand_distr::Dirichlet;

use kz_core::mapping::BoardMapper;
use kz_core::muzero::step::{
    muzero_step_apply, muzero_step_gather, MuZeroExpandRequest, MuZeroRequest, MuZeroResponse, MuZeroRootRequest,
};
use kz_core::muzero::tree::MuTree;
use kz_core::muzero::MuZeroEvaluation;
use kz_core::network::common::normalize_in_place;
use kz_core::network::muzero::{EvalResponsePair, ExpandArgs};
use kz_core::network::ZeroEvaluation;
use kz_core::zero::step::FpuMode;

use crate::move_selector::MoveSelector;
use crate::server::job_channel::JobClient;
use crate::server::protocol::{GeneratorUpdate, Settings};
use crate::simulation::{Position, Simulation};

type RootClient<B> = JobClient<B, EvalResponsePair>;
type ExpandClient = JobClient<ExpandArgs, EvalResponsePair>;

type UpdateSender<B> = crossbeam::channel::Sender<GeneratorUpdate<B>>;
type RngType = ThreadRng;

//TODO support max moves somehow?
pub fn generator_muzero_main<B: Board>(
    thread_id: usize,
    mapper: impl BoardMapper<B>,
    start_pos: impl Fn() -> B,
    batch_size: usize,
    settings_receiver: Receiver<Settings>,
    root_client: RootClient<B>,
    expand_client: ExpandClient,
    update_sender: UpdateSender<B>,
) -> Result<(), SendError<GeneratorUpdate<B>>> {
    let mut state = GeneratorState::new();

    let mut rng = RngType::default();

    let mut settings = None;
    let mut next_index = 0;

    loop {
        // if we don't have settings yet, block
        // TODO maybe add this to the big select lower down?
        //   meh, this should have higher priority than others
        let new_settings = if settings.is_some() {
            settings_receiver.try_recv()
        } else {
            settings_receiver.recv().map_err(|_| TryRecvError::Disconnected)
        };

        match new_settings {
            Ok(new_settings) => settings = Some(new_settings),
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => panic!("Settings channel disconnected"),
        }

        if let Some(settings) = &settings {
            let mut counter = Counter::default();

            let mut ctx = Context {
                thread_id,
                mapper,
                start_pos: &start_pos,
                batch_size,

                settings,
                root_client: &root_client,
                expand_client: &expand_client,
                update_sender: &update_sender,

                rng: &mut rng,
                next_index: &mut next_index,
                counter: &mut counter,
            };

            state.step(&mut ctx);

            // report progress
            update_sender.send(GeneratorUpdate::Progress {
                cached_evals: 0,
                root_evals: counter.root_evals,
                real_evals: counter.expand_evals,
                moves: counter.move_count,
            })?;
        }
    }
}

#[derive(Debug)]
struct Context<'a, B: Board, M, F> {
    thread_id: usize,
    mapper: M,
    start_pos: &'a F,
    // TODO rename this to something else, we're not really batching any more
    batch_size: usize,

    settings: &'a Settings,
    root_client: &'a RootClient<B>,
    expand_client: &'a ExpandClient,
    update_sender: &'a UpdateSender<B>,

    rng: &'a mut RngType,
    next_index: &'a mut u64,
    counter: &'a mut Counter,
}

#[derive(Debug)]
struct GeneratorState<B: Board, M> {
    games: Vec<GameTuple<B, M>>,
}

#[derive(Debug)]
struct GameTuple<B: Board, M> {
    state: GameState<B, M>,
    receiver: Receiver<EvalResponsePair>,
    node: usize,
}

#[derive(Debug)]
struct GameState<B: Board, M> {
    index: u64,
    search: SearchState<B, M>,
    mv_count: u32,
    positions: Vec<Position<B>>,
}

#[derive(Debug)]
struct SearchState<B, M> {
    tree: MuTree<B, M>,
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
    root_evals: u64,
    expand_evals: u64,
    move_count: u64,
}

impl<B: Board, M: BoardMapper<B>> GeneratorState<B, M> {
    fn new() -> Self {
        GeneratorState { games: vec![] }
    }

    fn step_and_append<F: Fn() -> B>(
        &mut self,
        ctx: &mut Context<B, M, F>,
        mut state: GameState<B, M>,
        response: Option<MuZeroResponse>,
    ) {
        let result = state.step(ctx, response);

        match result {
            StepResult::Done => {}
            StepResult::Request(request) => {
                let (node, receiver) = match request {
                    MuZeroRequest::Root(MuZeroRootRequest { node, board }) => {
                        ctx.counter.root_evals += 1;
                        (node, ctx.root_client.map(board))
                    }
                    MuZeroRequest::Expand(MuZeroExpandRequest {
                        node,
                        state,
                        move_index,
                    }) => {
                        ctx.counter.expand_evals += 1;
                        (node, ctx.expand_client.map((state, move_index)))
                    }
                };

                self.games.push(GameTuple { state, receiver, node })
            }
        }
    }

    fn step<F: Fn() -> B>(&mut self, ctx: &mut Context<B, M, F>) {
        // start new games until we have enough of them
        // do this before waiting for receivers so we don't block on an empty list
        let mut started_any = false;
        while self.games.len() < ctx.batch_size {
            let state = GameState::new(ctx);
            self.step_and_append(ctx, state, None);
            started_any = true;
        }

        if started_any {
            ctx.update_sender
                .send(GeneratorUpdate::StartedSimulations {
                    thread_id: ctx.thread_id,
                    next_index: *ctx.next_index,
                })
                .unwrap();
        }

        // wait for any receiver to become ready
        let mut select = Select::new();
        for t in &self.games {
            select.recv(&t.receiver);
        }
        select.ready();

        // update all existing games
        // this happens in bulk so we don't waste time repeatedly selecting on all receivers
        // TODO this could just use retain_mut when that becomes stable
        let prev_tuples = std::mem::take(&mut self.games);

        for tuple in prev_tuples {
            let response_pair = match tuple.receiver.try_recv() {
                Ok(response_pair) => response_pair,
                Err(TryRecvError::Empty) => {
                    // no luck, just put it back into the queue
                    self.games.push(tuple);
                    continue;
                }
                Err(TryRecvError::Disconnected) => panic!("Receiver disconnected"),
            };

            let response = MuZeroResponse {
                node: tuple.node,
                state: response_pair.0,
                eval: response_pair.1,
            };

            self.step_and_append(ctx, tuple.state, Some(response));
        }
    }
}

impl<B: Board, M: BoardMapper<B>> GameState<B, M> {
    fn new<F: Fn() -> B>(ctx: &mut Context<B, M, F>) -> Self {
        let start_pos = (ctx.start_pos)();
        let tree = MuTree::new(start_pos, ctx.mapper);

        let index = *ctx.next_index;
        *ctx.next_index += 1;

        GameState {
            index,
            mv_count: 0,
            search: SearchState::new(ctx, tree),
            positions: vec![],
        }
    }

    fn step<F: Fn() -> B>(
        &mut self,
        ctx: &mut Context<B, M, F>,
        initial_response: Option<MuZeroResponse>,
    ) -> StepResult<MuZeroRequest<B>> {
        let mut response = initial_response;

        loop {
            let result = self.search.step(ctx, self.mv_count, response.take());

            match result {
                StepResult::Request(request) => {
                    return StepResult::Request(request);
                }
                StepResult::Done => {
                    ctx.counter.move_count += 1;
                    if self.search_done_step(ctx) {
                        return StepResult::Done;
                    }
                }
            }
        }
    }

    fn search_done_step<F>(&mut self, ctx: &mut Context<B, M, F>) -> bool {
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
            is_full_search: self.search.is_full_search,
            played_mv: picked_move,
            zero_visits: tree.root_visits(),
            net_evaluation,
            zero_evaluation,
        });

        let mut next_board = tree.root_board().clone();
        next_board.play(picked_move);
        self.mv_count += 1;

        if next_board.is_done() || self.mv_count >= ctx.settings.max_game_length as u32 {
            // record this game
            let simulation = Simulation {
                positions: std::mem::take(&mut self.positions),
                final_board: next_board,
            };
            ctx.update_sender
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
            let next_tree = MuTree::new(next_board, ctx.mapper);
            self.search = SearchState::new(ctx, next_tree);

            // report that this game is not done
            false
        }
    }
}

impl<B: Board, M: BoardMapper<B>> SearchState<B, M> {
    fn new<F>(ctx: &mut Context<B, M, F>, tree: MuTree<B, M>) -> Self {
        SearchState {
            tree,
            is_full_search: ctx.rng.gen_bool(ctx.settings.full_search_prob),
            root_net_eval: None,
        }
    }

    fn step<F>(
        &mut self,
        ctx: &mut Context<B, M, F>,
        mv_count: u32,
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
            muzero_step_apply(&mut self.tree, top_moves, response);
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

            let draw_depth = ctx.settings.max_game_length as u32 - mv_count;

            if let Some(request) = muzero_step_gather(
                &mut self.tree,
                settings.weights.to_uct(),
                settings.use_value,
                FpuMode::Parent,
                draw_depth,
            ) {
                return StepResult::Request(request);
            }
        }
    }
}

fn add_dirichlet_noise<B: Board, M: BoardMapper<B>, F>(ctx: &mut Context<B, M, F>, board: &B, policy: &mut [f32]) {
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
