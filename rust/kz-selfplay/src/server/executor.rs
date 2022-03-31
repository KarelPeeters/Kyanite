use crate::server::job_channel::{Job, JobServer};
use board_game::board::Board;
use crossbeam::channel::Receiver;
use crossbeam::select;
use cuda_sys::wrapper::handle::Device;
use kz_core::mapping::BoardMapper;
use kz_core::network::muzero::{EvalResponsePair, ExpandArgs, MuZeroFusedGraphs};
use std::sync::Arc;

#[derive(Debug)]
enum ExecutorMessage<G, X, Y> {
    Graph(G),
    Job(Job<X, Y>),
}

fn receiver_executor_message<G, X, Y>(
    graph_receiver: &Receiver<G>,
    server: &JobServer<X, Y>,
) -> ExecutorMessage<G, X, Y> {
    select! {
        recv(graph_receiver) -> graph => ExecutorMessage::Graph(graph.unwrap()),
        recv(server.receiver()) -> job => ExecutorMessage::Job(job.unwrap()),
    }
}

pub fn executor_loop_root<B: Board, M: BoardMapper<B>>(
    device: Device,
    batch_size: usize,
    graph_receiver: Receiver<Arc<MuZeroFusedGraphs<B, M>>>,
    server: JobServer<Vec<B>, Vec<EvalResponsePair>>,
) {
    // wait for the initial graph
    let graph = graph_receiver.recv().unwrap();
    let mut executor = graph.root_executor(device, batch_size);

    // handle incoming messages
    loop {
        match receiver_executor_message(&graph_receiver, &server) {
            ExecutorMessage::Graph(graph) => {
                drop(executor);
                executor = graph.root_executor(device, batch_size);
            }
            ExecutorMessage::Job(job) => {
                job.run(|x| executor.eval_root(&x));
            }
        }
    }
}

pub fn executor_loop_expander<B: Board, M: BoardMapper<B>>(
    device: Device,
    batch_size: usize,
    graph_receiver: Receiver<Arc<MuZeroFusedGraphs<B, M>>>,
    server: JobServer<Vec<ExpandArgs>, Vec<EvalResponsePair>>,
) {
    // wait for the initial graph
    let graph = graph_receiver.recv().unwrap();
    let mut executor = graph.expand_executor(device, batch_size);

    // handle incoming messages
    loop {
        match receiver_executor_message(&graph_receiver, &server) {
            ExecutorMessage::Graph(graph) => {
                drop(executor);
                executor = graph.expand_executor(device, batch_size);
            }
            ExecutorMessage::Job(job) => {
                job.run(|x| executor.eval_expand(&x));
            }
        }
    }
}
