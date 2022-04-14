use std::sync::Arc;

use board_game::board::Board;
use crossbeam::channel;
use crossbeam::channel::Sender;
use crossbeam::thread::Scope;

use cuda_sys::wrapper::handle::Device;
use kz_core::mapping::BoardMapper;
use nn_graph::graph::Graph;
use nn_graph::onnx::load_graph_from_onnx_path;
use nn_graph::optimizer::optimize_graph;

use crate::server::executor::executor_loop_alphazero;
use crate::server::generator_alphazero::generator_alphazero_main;
use crate::server::protocol::{GeneratorUpdate, Settings, StartupSettings};
use crate::server::rebatcher::Rebatcher;
use crate::server::server::ZeroSpecialization;

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
        start_pos: &'s (impl Fn() -> B + Sync),
        update_sender: Sender<GeneratorUpdate<B>>,
    ) -> (Vec<Sender<Settings>>, Vec<Sender<Arc<Graph>>>) {
        let cpu_batch_size = startup.cpu_batch_size;
        let gpu_batch_size = startup.gpu_batch_size;

        let mut settings_senders: Vec<Sender<Settings>> = vec![];
        let mut graph_senders: Vec<Sender<Arc<Graph>>> = vec![];

        let (rebatcher, eval_client, eval_server) = Rebatcher::new(2, startup.cpu_batch_size, gpu_batch_size);

        // spawn cpu threads
        for local_id in 0..startup.cpu_threads_per_device {
            let thread_id = startup.cpu_threads_per_device * device_id + local_id;

            let eval_client = eval_client.clone();
            let update_sender = update_sender.clone();

            let (settings_sender, settings_receiver) = channel::bounded(1);
            settings_senders.push(settings_sender);

            s.builder()
                .name(format!("generator-{}-{}", device_id, local_id))
                .spawn(move |_| {
                    generator_alphazero_main(
                        thread_id,
                        start_pos,
                        cpu_batch_size,
                        settings_receiver,
                        eval_client,
                        update_sender,
                    )
                })
                .unwrap();
        }

        // spawn gpu eval threads
        for local_id in 0..startup.gpu_threads_per_device {
            let (graph_sender, graph_receiver) = channel::bounded(1);
            graph_senders.push(graph_sender);

            let eval_server = eval_server.clone();

            s.builder()
                .name(format!("gpu-expand-{}-{}", device_id, local_id))
                .spawn(move |_| {
                    executor_loop_alphazero(device, gpu_batch_size, mapper, graph_receiver, eval_server);
                })
                .unwrap();
        }

        // spawn rebatcher thread
        s.builder()
            .name(format!("rebatcher-{}", device_id))
            .spawn(move |_| {
                rebatcher.run_loop();
            })
            .unwrap();

        (settings_senders, graph_senders)
    }

    fn load_graph(&self, path: &str, _: M) -> Self::G {
        optimize_graph(&load_graph_from_onnx_path(path), Default::default())
    }
}
