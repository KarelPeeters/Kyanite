use std::cmp::min;

use board_game::board::Board;
use flume::Sender;
use futures::executor::ThreadPoolBuilder;
use itertools::Itertools;
use rand::rngs::StdRng;
use rand::SeedableRng;

use cuda_sys::wrapper::handle::Device;
use kz_core::mapping::BoardMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::job_channel::job_pair;
use kz_core::network::{EvalClient, Network};
use kz_core::zero::values::ZeroValuesPov;
use kz_core::zero::wrapper::ZeroSettings;
use kz_selfplay::server::executor::{batched_executor_loop, RunCondition};
use kz_util::math::ceil_div;
use kz_util::throughput::PrintThroughput;
use nn_graph::graph::Graph;

#[derive(Debug, Copy, Clone)]
pub struct BatchEvalSettings {
    pub visits: u64,
    pub network_batch_size: usize,
    pub cpu_threads: usize,
    pub max_concurrent_positions: usize,
}

#[derive(Debug)]
struct PositionEval {
    pi: usize,
    eval: ZeroValuesPov,
}

pub fn batch_tree_eval<B: Board>(
    positions: Vec<B>,
    settings: BatchEvalSettings,
    zero_settings: ZeroSettings,
    graph: Graph,
    mapper: impl BoardMapper<B>,
    device: Device,
) -> Vec<ZeroValuesPov> {
    let position_count = positions.len();

    println!("Evaluating {} positions", position_count);
    let approx_evals_per_pos = ceil_div(settings.visits as usize, zero_settings.batch_size) * zero_settings.batch_size;
    let approx_evals_total = approx_evals_per_pos * positions.len();

    let (job_client, job_server) = job_pair(2 * ceil_div(settings.network_batch_size, zero_settings.batch_size));
    let (result_sender, result_receiver) = flume::unbounded();

    let pool = ThreadPoolBuilder::new()
        .name_prefix("pool")
        .pool_size(settings.cpu_threads)
        .create()
        .unwrap();

    let captured_positions = positions.clone();
    let spawn_position = move |pi: usize, job_client: EvalClient<_>, result_sender: Sender<PositionEval>| {
        println!("Spawning {}", pi);
        let position = captured_positions[pi].clone();

        pool.spawn_ok(async move {
            let mut rng = StdRng::from_entropy();

            let tree = zero_settings
                .build_tree_async(&position, &job_client, &mut rng, |tree| {
                    tree.root_visits() >= settings.visits
                })
                .await;

            result_sender
                .send(PositionEval {
                    pi,
                    eval: tree.values(),
                })
                .unwrap();
        })
    };

    let initial_positions = min(positions.len(), settings.max_concurrent_positions);
    for pi in 0..initial_positions {
        spawn_position(pi, job_client.clone(), result_sender.clone());
    }

    crossbeam::scope(move |s| {
        s.builder()
            .name("executor".into())
            .spawn(move |_| {
                let mut tp = PrintThroughput::new("evals");

                let mut total_filled: u64 = 0;
                let mut delta_filled: u64 = 0;
                let mut delta_potential: u64 = 0;

                let (graph_sender, graph_receiver) = flume::bounded(1);
                graph_sender.send(Some(graph)).unwrap();
                drop(graph_sender);

                let network_batch_size = settings.network_batch_size;
                batched_executor_loop(
                    network_batch_size,
                    RunCondition::Any,
                    graph_receiver,
                    job_server,
                    |graph| CudaNetwork::new(mapper, &graph, network_batch_size, device),
                    move |network, batch_x| {
                        let result = network.evaluate_batch(&batch_x);

                        total_filled += batch_x.len() as u64;
                        delta_filled += batch_x.len() as u64;
                        delta_potential += network_batch_size as u64;

                        if tp.update_delta(network_batch_size as u64) {
                            println!("  fill rate: {}", delta_filled as f32 / delta_potential as f32);
                            println!("  progress: ~{}", total_filled as f32 / approx_evals_total as f32);
                            delta_filled = 0;
                            delta_potential = 0;
                        }

                        result
                    },
                );
            })
            .unwrap();

        let result = s
            .builder()
            .name("collector".into())
            .spawn(move |_| {
                let mut received = 0;
                let mut results = vec![None; position_count];

                let mut job_client = Some(job_client);
                let mut result_sender = Some(result_sender);

                for result in result_receiver {
                    received += 1;

                    assert!(
                        results[result.pi].is_none(),
                        "Received duplicate position {}",
                        result.pi
                    );
                    results[result.pi] = Some(result.eval);

                    let next_spawn_index = initial_positions + received - 1;
                    if next_spawn_index < position_count {
                        let job_client = job_client.as_ref().unwrap().clone();
                        let result_sender = result_sender.as_ref().unwrap().clone();
                        spawn_position(next_spawn_index, job_client, result_sender);
                    } else {
                        // drop so that executor (and this for loop) will stop
                        job_client.take();
                        result_sender.take();
                    }

                    println!("Done {}/{} positions", received, position_count);
                }

                let result = results.into_iter().map(Option::unwrap).collect_vec();
                result
            })
            .unwrap();

        result.join().unwrap()
    })
    .unwrap()
}
