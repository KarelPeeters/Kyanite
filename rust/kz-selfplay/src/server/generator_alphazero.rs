use board_game::board::Board;
use board_game::games::max_length::MaxMovesBoard;
use flume::{Receiver, TryRecvError};
use lru::LruCache;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Dirichlet;

use kz_core::network::ZeroEvaluation;
use kz_core::oracle::DummyOracle;
use kz_core::zero::step::{zero_step_apply, zero_step_gather, FpuMode};
use kz_core::zero::tree::Tree;
use kz_util::sequence::zip_eq_exact;

use crate::move_selector::MoveSelector;
use crate::server::job_channel::JobClient;
use crate::server::protocol::{GeneratorUpdate, Settings};
use crate::server::server::UpdateSender;
use crate::simulation::{Position, Simulation};

type EvalClient<B> = JobClient<B, ZeroEvaluation<'static>>;

pub async fn generator_alphazero_main<B: Board>(
    generator_id: usize,
    start_pos: impl Fn() -> B,
    settings_receiver: Receiver<Settings>,
    eval_client: EvalClient<B>,
    update_sender: UpdateSender<B>,
) {
    // wait for initial settings
    let mut settings = settings_receiver.recv_async().await.unwrap();

    //TODO try with a different(faster) rng
    //  really? can the rng be that significant?
    let mut rng = StdRng::from_entropy();

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

async fn generate_simulation<B: Board>(
    generator_id: usize,
    settings: &Settings,
    update_sender: &UpdateSender<B>,
    eval_client: &EvalClient<B>,
    start: B,
    rng: &mut impl Rng,
) -> Simulation<B> {
    // create a new cache for every game, to prevent long-term stale values for short games
    // TODO maybe explicitly clear the cache when a new network is loaded instead?
    let mut cache: LruCache<B, ZeroEvaluation> = LruCache::new(settings.cache_size);

    let mut positions = vec![];

    let max_moves = settings.max_game_length.unwrap_or(u64::MAX);
    let mut curr_board = MaxMovesBoard::new(start, max_moves);

    while !curr_board.is_done() {
        // update stats to collect
        let mut cached_evals = 0;

        // determinate search settings
        let is_full_search = rng.gen_bool(settings.full_search_prob);
        let target_visits = if is_full_search {
            settings.full_iterations
        } else {
            settings.part_iterations
        };

        // run tree search
        let mut tree = Tree::new(curr_board.clone());
        let mut root_net_eval = None;

        while tree.root_visits() < target_visits {
            // TODO add oracle support (once that actually works)
            let request = zero_step_gather(
                &mut tree,
                &DummyOracle,
                settings.weights.to_uct(),
                settings.use_value,
                FpuMode::Parent,
            );
            if let Some(request) = request {
                let board = request.board.inner();

                let mut eval: ZeroEvaluation = if let Some(eval) = cache.get(board) {
                    cached_evals += 1;
                    eval.clone()
                } else {
                    let eval = eval_client.map_async(board.clone()).await;
                    cache.put(board.clone(), eval.clone());
                    eval
                };

                // add dirichlet noise to the evaluation if this is the root eval
                if root_net_eval.is_none() {
                    root_net_eval = Some(eval.clone());
                    add_dirichlet_noise(eval.policy.to_mut(), settings, rng);
                }

                let response = request.respond(eval);
                zero_step_apply(&mut tree, response);
            }
        }

        // extract stats
        let net_evaluation = root_net_eval.unwrap();
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
        if cached_evals != 0 {
            update_sender
                .send(GeneratorUpdate::Evals {
                    cached_evals: cached_evals,
                    real_evals: 0,
                    root_evals: 0,
                })
                .unwrap();
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
