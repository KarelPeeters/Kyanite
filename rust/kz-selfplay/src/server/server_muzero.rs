use crate::server::executor::batched_executor_loop;
use crate::server::generator_muzero::generator_muzero_main;
use crate::server::job_channel::job_pair;
use crate::server::protocol::{GeneratorUpdate, Settings, StartupSettings};
use crate::server::server::{GraphSender, ZeroSpecialization};
use board_game::board::Board;
use crossbeam::thread::Scope;
use cuda_sys::wrapper::handle::Device;
use flume::Sender;
use futures::executor::ThreadPoolBuilder;
use kz_core::mapping::BoardMapper;
use kz_core::network::muzero::{MuZeroFusedGraphs, MuZeroGraphs};

#[derive(Debug)]
pub struct MuZeroSpecialization;

impl<B: Board, M: BoardMapper<B> + 'static> ZeroSpecialization<B, M> for MuZeroSpecialization {
    type G = MuZeroFusedGraphs<B, M>;

    fn spawn_device_threads<'s>(
        &self,
        s: &Scope<'s>,
        device: Device,
        device_id: usize,
        startup: &StartupSettings,
        mapper: M,
        start_pos: impl Fn() -> B + Sync + Send + Clone + 'static,
        update_sender: Sender<GeneratorUpdate<B>>,
    ) -> (Vec<Sender<Settings>>, Vec<GraphSender<Self::G>>) {
        let gpu_batch_size_root = startup.gpu_batch_size_root;
        let gpu_batch_size_expand = startup.gpu_batch_size;
        let cpu_threads = startup.cpu_threads_per_device;
        let gpu_threads = startup.gpu_threads_per_device;
        let concurrent_games = (gpu_threads + 1) * gpu_batch_size_expand + 2 * gpu_batch_size_expand;
        println!("Running {} concurrent games", concurrent_games);

        let mut settings_senders: Vec<Sender<Settings>> = vec![];
        let mut graph_senders: Vec<GraphSender<Self::G>> = vec![];

        // TODO is it worth it to have a rebatcher again? it might take some CPU load from the GPU thread
        let (root_client, root_server) = job_pair(gpu_batch_size_root);
        let (expand_client, expand_server) = job_pair(gpu_batch_size_expand);

        // spawn cpu threads
        let pool = ThreadPoolBuilder::new()
            .pool_size(cpu_threads)
            .name_prefix(format!("generator-{}-", device_id))
            .create()
            .unwrap();

        for generator_id in 0..concurrent_games {
            let start_pos = start_pos.clone();
            let root_client = root_client.clone();
            let expand_client = expand_client.clone();
            let update_sender = update_sender.clone();

            let (settings_sender, settings_receiver) = flume::bounded(1);
            settings_senders.push(settings_sender);

            pool.spawn_ok(async move {
                generator_muzero_main(
                    generator_id,
                    start_pos,
                    mapper,
                    settings_receiver,
                    root_client,
                    expand_client,
                    update_sender,
                )
                .await;
            });
        }

        // spawn gpu expand eval threads
        for local_id in 0..gpu_threads {
            let (graph_sender, graph_receiver) = flume::bounded(1);
            graph_senders.push(graph_sender);

            let expand_server = expand_server.clone();

            s.builder()
                .name(format!("gpu-expand-{}-{}", device_id, local_id))
                .spawn(move |_| {
                    batched_executor_loop(
                        gpu_batch_size_expand,
                        graph_receiver,
                        expand_server,
                        |graph| graph.expand_executor(device, gpu_batch_size_expand),
                        |network, x| network.eval_expand(&x),
                    );
                })
                .unwrap();
        }

        // spawn gpu root eval thread
        {
            let (graph_sender, graph_receiver) = flume::bounded(1);
            graph_senders.push(graph_sender);

            let root_server = root_server.clone();

            s.builder()
                .name(format!("gpu-root-{}", device_id))
                .spawn(move |_| {
                    batched_executor_loop(
                        gpu_batch_size_root,
                        graph_receiver,
                        root_server,
                        |graph| graph.root_executor(device, gpu_batch_size_root),
                        |network, x| network.eval_root(&x),
                    );
                })
                .unwrap();
        }

        (settings_senders, graph_senders)
    }

    fn load_graph(&self, path: &str, mapper: M) -> Self::G {
        MuZeroGraphs::load(path, mapper)
            .optimize(Default::default())
            .fuse(Default::default())
    }
}
