use crate::server::executor::{executor_loop_expander, executor_loop_root};
use crate::server::generator_muzero::generator_muzero_main;
use crate::server::job_channel::job_pair;
use crate::server::protocol::{GeneratorUpdate, Settings, StartupSettings};
use crate::server::server::ZeroSpecialization;
use board_game::board::Board;
use crossbeam::channel;
use crossbeam::channel::Sender;
use crossbeam::thread::Scope;
use cuda_sys::wrapper::handle::Device;
use kz_core::mapping::BoardMapper;
use kz_core::network::muzero::{MuZeroFusedGraphs, MuZeroGraphs};
use std::sync::Arc;

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
        start_pos: &'s (impl Fn() -> B + Sync),
        update_sender: Sender<GeneratorUpdate<B>>,
    ) -> (Vec<Sender<Settings>>, Vec<Sender<Arc<Self::G>>>) {
        let cpu_batch_size = startup.cpu_batch_size;
        let gpu_batch_size_expand = startup.gpu_batch_size;
        let gpu_batch_size_root = startup.gpu_batch_size_root;

        let mut settings_senders: Vec<Sender<Settings>> = vec![];
        let mut graph_senders: Vec<Sender<Arc<MuZeroFusedGraphs<B, M>>>> = vec![];

        // TODO is it worth it to have a rebatcher again? it might take some CPU load from the GPU thread
        let (root_client, root_server) = job_pair(gpu_batch_size_root);
        let (expand_client, expand_server) = job_pair(gpu_batch_size_expand);

        // spawn cpu threads
        for local_id in 0..startup.cpu_threads_per_device {
            let thread_id = startup.cpu_threads_per_device * device_id + local_id;

            let root_client = root_client.clone();
            let expand_client = expand_client.clone();
            let update_sender = update_sender.clone();

            let (settings_sender, settings_receiver) = channel::bounded(1);
            settings_senders.push(settings_sender);

            s.builder()
                .name(format!("generator-{}-{}", device_id, local_id))
                .spawn(move |_| {
                    generator_muzero_main(
                        thread_id,
                        mapper,
                        start_pos,
                        cpu_batch_size,
                        settings_receiver,
                        root_client,
                        expand_client,
                        update_sender,
                    )
                })
                .unwrap();
        }

        // spawn gpu expand eval threads
        for local_id in 0..startup.gpu_threads_per_device {
            let (graph_sender, graph_receiver) = channel::bounded(1);
            graph_senders.push(graph_sender);

            let expand_server = expand_server.clone();

            s.builder()
                .name(format!("gpu-expand-{}-{}", device_id, local_id))
                .spawn(move |_| {
                    executor_loop_expander(device, gpu_batch_size_expand, graph_receiver, expand_server);
                })
                .unwrap();
        }

        // spawn gpu root eval thread
        {
            let (graph_sender, graph_receiver) = channel::bounded(1);
            graph_senders.push(graph_sender);

            let root_server = root_server.clone();

            s.builder()
                .name(format!("gpu-root-{}", device_id))
                .spawn(move |_| {
                    executor_loop_root(device, gpu_batch_size_root, graph_receiver, root_server);
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
