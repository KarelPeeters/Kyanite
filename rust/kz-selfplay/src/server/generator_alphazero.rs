use board_game::board::Board;
use board_game::games::max_length::MaxMovesBoard;
use flume::{Receiver, TryRecvError};
use itertools::Itertools;
use lru::LruCache;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Dirichlet;

use kz_core::network::common::policy_softmax_temperature_in_place;
use kz_core::network::{EvalClient, ZeroEvaluation};
use kz_core::zero::step::{zero_step_apply, zero_step_gather, ZeroRequest};
use kz_core::zero::tree::Tree;
use kz_util::sequence::zip_eq_exact;

use crate::move_selector::MoveSelector;
use crate::server::protocol::{Evals, GeneratorUpdate, Settings};
use crate::server::server::UpdateSender;
use crate::simulation::{Position, Simulation};

pub async fn generator_alphazero_main<B: Board>(
    generator_id: usize,
    start_pos: impl Fn() -> B,
    settings_receiver: Receiver<Settings>,
    search_batch_size: usize,
    eval_client: EvalClient<B>,
    update_sender: UpdateSender<B>,
) {
    let mut rng = StdRng::from_entropy();

    // wait for initial settings
    let mut settings = settings_receiver.recv_async().await.unwrap();

    loop {
        // possibly get new settings
        match settings_receiver.try_recv() {
            Ok(new_settings) => settings = new_settings,
            Err(TryRecvError::Empty) => (),
            Err(TryRecvError::Disconnected) => break,
        }

        update_sender
            .send(GeneratorUpdate::StartedSimulation { generator_id })
            .unwrap();

        let simulation = generate_simulation(
            generator_id,
            &settings,
            search_batch_size,
            &update_sender,
            &eval_client,
            start_pos(),
            &mut rng,
        )
        .await;

        update_sender
            .send(GeneratorUpdate::FinishedSimulation {
                generator_id,
                simulation,
            })
            .unwrap();
    }
}

type Cache<B> = LruCache<B, ZeroEvaluation<'static>>;

async fn generate_simulation<B: Board>(
    generator_id: usize,
    settings: &Settings,
    search_batch_size: usize,
    update_sender: &UpdateSender<B>,
    eval_client: &EvalClient<B>,
    start: B,
    rng: &mut impl Rng,
) -> Simulation<B> {
    // create a new cache for every game, to prevent long-term stale values for short games
    // TODO maybe explicitly clear the cache when a new network is loaded instead?
    let mut cache: Cache<B> = LruCache::new(settings.cache_size);

    let mut positions = vec![];

    let max_moves = settings.max_game_length.unwrap_or(u64::MAX);
    let mut curr_board = MaxMovesBoard::new(start, max_moves);

    while !curr_board.is_done() {
        // determinate search settings
        let is_full_search = rng.gen_bool(settings.full_search_prob);
        let target_visits = if is_full_search {
            settings.full_iterations
        } else {
            settings.part_iterations
        };

        // run tree search
        let (tree, cached_evals, net_evaluation) = build_tree(
            settings,
            search_batch_size,
            eval_client,
            &mut cache,
            &curr_board,
            target_visits,
            rng,
        )
        .await;
        let zero_evaluation = tree.eval();

        // pick a move to play
        let move_selector = MoveSelector::new(settings.temperature, settings.zero_temp_move_count);
        let picked_index = move_selector.select(positions.len() as u32, zero_evaluation.policy.as_ref(), rng);
        let picked_child = tree[0].children.unwrap().get(picked_index);
        let picked_move = tree[picked_child].last_move.unwrap();

        // record position
        let position = Position {
            board: curr_board.inner().clone(),
            is_full_search,
            played_mv: picked_move,
            zero_visits: tree.root_visits(),
            zero_evaluation,
            net_evaluation,
        };
        positions.push(position);

        // actually play the move
        curr_board.play(picked_move);

        // send updates
        // TODO these cached evals are somewhat temporally misaligned with when the evals actually take place,
        //   can this cause issues when reporting things like cache hit rate?
        if cached_evals != 0 {
            let msg = GeneratorUpdate::ExpandEvals(Evals::new(0, 0, cached_evals));
            update_sender.send(msg).unwrap();
        }
        update_sender
            .send(GeneratorUpdate::FinishedMove {
                generator_id,
                curr_game_length: positions.len(),
            })
            .unwrap();
    }

    Simulation {
        positions,
        final_board: curr_board.into_inner(),
    }
}

async fn build_tree<B: Board>(
    settings: &Settings,
    search_batch_size: usize,
    eval_client: &EvalClient<B>,
    cache: &mut Cache<B>,
    curr_board: &MaxMovesBoard<B>,
    target_visits: u64,
    rng: &mut impl Rng,
) -> (Tree<MaxMovesBoard<B>>, usize, ZeroEvaluation<'static>) {
    let mut tree = Tree::new(curr_board.clone());
    let mut cached_evals = 0;
    let mut root_net_eval = None;

    while tree.root_visits() < target_visits {
        let mut requests = vec![];
        let mut terminal_gathers = 0;

        // collect a batch of requests
        while requests.len() < search_batch_size && terminal_gathers < search_batch_size {
            let request = zero_step_gather(
                &mut tree,
                settings.weights.to_uct(),
                settings.q_mode.0,
                settings.search_fpu_root.0,
                settings.search_fpu_child.0,
                settings.search_virtual_loss_weight,
                rng,
            );

            match request {
                Some(request) => {
                    let board = request.board.inner();

                    match cache.get(board) {
                        // TODO immediately applying the eval on cache hits could bias the search, is that a problem?
                        //   (for selfplay we usually use small batches sizes so it's not that bad)
                        Some(eval) => {
                            cached_evals += 1;
                            apply_eval(&mut tree, request, eval.clone(), &mut root_net_eval, settings, rng);
                        }
                        None => {
                            requests.push(request);
                        }
                    }
                }
                None => {
                    terminal_gathers += 1;
                }
            }
        }

        // evaluate requests
        let boards = requests.iter().map(|r| r.board.inner().clone()).collect_vec();
        let evals = eval_client.map_async(boards).await;

        // apply all of them
        for (request, eval) in zip_eq_exact(requests, evals) {
            cache.put(request.board.inner().clone(), eval.clone());
            apply_eval(&mut tree, request, eval, &mut root_net_eval, settings, rng);
        }
    }

    let net_evaluation = root_net_eval.unwrap();
    (tree, cached_evals, net_evaluation)
}

fn apply_eval<B: Board>(
    tree: &mut Tree<B>,
    request: ZeroRequest<B>,
    mut eval: ZeroEvaluation<'static>,
    root_net_eval: &mut Option<ZeroEvaluation>,
    settings: &Settings,
    rng: &mut impl Rng,
) {
    // record root eval
    if request.node == 0 {
        *root_net_eval = Some(eval.clone());
    }

    // policy softmax temperature
    let temperature = if request.node == 0 {
        settings.search_policy_temperature_root
    } else {
        settings.search_policy_temperature_child
    };
    policy_softmax_temperature_in_place(eval.policy.to_mut(), temperature);

    // dirichlet noise
    if request.node == 0 {
        add_dirichlet_noise(eval.policy.to_mut(), settings, rng);
    }

    // add to tree
    zero_step_apply(tree, request.respond(eval));
}

fn add_dirichlet_noise(policy: &mut [f32], settings: &Settings, rng: &mut impl Rng) {
    let alpha = settings.dirichlet_alpha;
    let eps = settings.dirichlet_eps;

    if policy.len() > 1 && eps != 0.0 {
        let distr = Dirichlet::new_with_size(alpha, policy.len()).unwrap();
        let noise = rng.sample(distr);

        for (p, n) in zip_eq_exact(policy, noise) {
            *p = (1.0 - eps) * (*p) + eps * n;
        }
    }
}
