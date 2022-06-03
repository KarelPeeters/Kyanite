use board_game::board::Board;
use crossbeam::thread::Scope;
use flume::Sender;
use futures::executor::ThreadPoolBuilder;

use cuda_sys::wrapper::handle::Device;
use kz_core::mapping::BoardMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::job_channel::job_pair;
use kz_core::network::Network;
use kz_util::math::ceil_div;
use nn_graph::graph::Graph;
use nn_graph::onnx::load_graph_from_onnx_path;
use nn_graph::optimizer::optimize_graph;

use crate::server::executor::{batched_executor_loop, RunCondition};
use crate::server::generator_alphazero::generator_alphazero_main;
use crate::server::protocol::{Evals, GeneratorUpdate, Settings, StartupSettings};
use crate::server::server::{GraphSender, ZeroSpecialization};

#[derive(Debug)]
pub struct AlphaZeroSpecialization;

impl<B: Board, M: BoardMapper<B> + 'static> ZeroSpecialization<B, M> for AlphaZeroSpecialization {
    type G = Graph;

    fn spawn_device_threads<'s>(
        &self,
        s: &Scope<'s>,
        device: Device,
        device_id: usize,
        startup: &StartupSettings,
        mapper: M,
        start_pos: impl Fn() -> B + Send + Sync + Clone + 'static,
        update_sender: Sender<GeneratorUpdate<B>>,
    ) -> (Vec<Sender<Settings>>, Vec<GraphSender<Graph>>) {
        let gpu_batch_size = startup.gpu_batch_size;
        let search_batch_size = startup.search_batch_size;
        let cpu_threads = startup.cpu_threads_per_device;
        let gpu_threads = startup.gpu_threads_per_device;
        let concurrent_games = ceil_div((gpu_threads + 1) * gpu_batch_size, search_batch_size);
        println!("Spawning {} games", concurrent_games);

        let mut settings_senders: Vec<Sender<Settings>> = vec![];
        let mut graph_senders: Vec<GraphSender<Graph>> = vec![];

        let job_buffer_size = ceil_div(gpu_threads * gpu_batch_size, search_batch_size);
        let (eval_client, eval_server) = job_pair(job_buffer_size);

        // spawn cpu threads
        let pool = ThreadPoolBuilder::new()
            .pool_size(cpu_threads)
            .name_prefix(format!("generator-{}-", device_id))
            .create()
            .unwrap();

        for local_generator_id in 0..concurrent_games {
            let generator_id = concurrent_games * device_id + local_generator_id;

            let start_pos = start_pos.clone();
            let eval_client = eval_client.clone();
            let update_sender = update_sender.clone();

            let (settings_sender, settings_receiver) = flume::bounded(1);
            settings_senders.push(settings_sender);

            pool.spawn_ok(async move {
                generator_alphazero_main(
                    generator_id,
                    start_pos,
                    settings_receiver,
                    search_batch_size,
                    eval_client,
                    update_sender,
                )
                .await;
            });
        }

        // spawn gpu eval threads
        for local_id in 0..gpu_threads {
            let (graph_sender, graph_receiver) = flume::bounded(1);
            graph_senders.push(graph_sender);

            let eval_server = eval_server.clone();
            let update_sender = update_sender.clone();

            s.builder()
                .name(format!("gpu-expand-{}-{}", device_id, local_id))
                .spawn(move |_| {
                    batched_executor_loop(
                        gpu_batch_size,
                        RunCondition::FullBatch,
                        graph_receiver,
                        eval_server,
                        |graph| CudaNetwork::new(mapper, &graph, gpu_batch_size, device),
                        |network, x| {
                            let y = network.evaluate_batch(&x);
                            let msg = GeneratorUpdate::ExpandEvals(Evals::new(x.len(), gpu_batch_size, 0));
                            update_sender.send(msg).unwrap();
                            y
                        },
                    );
                })
                .unwrap();
        }

        (settings_senders, graph_senders)
    }

    fn load_graph(&self, path: &str, _: M, _: &StartupSettings) -> Self::G {
        optimize_graph(&load_graph_from_onnx_path(path), Default::default())
    }
}
