use std::sync::Arc;

use board_game::board::Board;
use flume::{Receiver, Selector};

use cuda_sys::wrapper::handle::Device;
use kz_core::mapping::BoardMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::muzero::{EvalResponsePair, ExpandArgs, MuZeroFusedGraphs};
use kz_core::network::{Network, ZeroEvaluation};
use kz_util::zip_eq_exact;
use nn_graph::graph::Graph;

use crate::server::job_channel::{Job, JobServer};

#[derive(Debug)]
enum ExecutorMessage<G, X, Y> {
    Graph(G),
    Job(Job<X, Y>),
}

fn receiver_executor_message<G, X, Y>(
    graph_receiver: &Receiver<G>,
    server: &JobServer<X, Y>,
) -> ExecutorMessage<G, X, Y> {
    Selector::new()
        .recv(graph_receiver, |graph| ExecutorMessage::Graph(graph.unwrap()))
        .recv(server.receiver(), |job| ExecutorMessage::Job(job.unwrap()))
        .wait()
}

pub fn executor_loop_alphazero<B: Board, M: BoardMapper<B>>(
    device: Device,
    batch_size: usize,
    mapper: M,
    graph_receiver: Receiver<Arc<Graph>>,
    server: JobServer<B, ZeroEvaluation>,
) {
    // wait for the initial graph
    let graph = graph_receiver.recv().unwrap();
    println!("{} loading new network", std::thread::current().name().unwrap());
    let mut network = CudaNetwork::new(mapper, &graph, batch_size, device);

    // handle incoming messages
    loop {
        match receiver_executor_message(&graph_receiver, &server) {
            ExecutorMessage::Graph(graph) => {
                println!("{} loading new network", std::thread::current().name().unwrap());
                drop(network);
                network = CudaNetwork::new(mapper, &graph, batch_size, device);
            }
            ExecutorMessage::Job(job) => {
                run_job_batch(batch_size, job, server.receiver(), |x| network.evaluate_batch(&x));
            }
        }
    }
}

pub fn executor_loop_root<B: Board, M: BoardMapper<B>>(
    device: Device,
    batch_size: usize,
    graph_receiver: Receiver<Arc<MuZeroFusedGraphs<B, M>>>,
    server: JobServer<B, EvalResponsePair>,
) {
    // wait for the initial graph
    let graph = graph_receiver.recv().unwrap();
    println!("{} loading new network", std::thread::current().name().unwrap());
    let mut executor = graph.root_executor(device, batch_size);

    // handle incoming messages
    loop {
        match receiver_executor_message(&graph_receiver, &server) {
            ExecutorMessage::Graph(graph) => {
                println!("{} loading new network", std::thread::current().name().unwrap());
                drop(executor);
                executor = graph.root_executor(device, batch_size);
            }
            ExecutorMessage::Job(job) => run_job_batch(batch_size, job, server.receiver(), |x| executor.eval_root(&x)),
        }
    }
}

pub fn executor_loop_expander<B: Board, M: BoardMapper<B>>(
    device: Device,
    batch_size: usize,
    graph_receiver: Receiver<Arc<MuZeroFusedGraphs<B, M>>>,
    server: JobServer<ExpandArgs, EvalResponsePair>,
) {
    // wait for the initial graph
    let graph = graph_receiver.recv().unwrap();
    println!("{} loading new network", std::thread::current().name().unwrap());
    let mut executor = graph.expand_executor(device, batch_size);

    // handle incoming messages
    loop {
        match receiver_executor_message(&graph_receiver, &server) {
            ExecutorMessage::Graph(graph) => {
                println!("{} loading new network", std::thread::current().name().unwrap());
                drop(executor);
                executor = graph.expand_executor(device, batch_size);
            }
            ExecutorMessage::Job(job) => {
                run_job_batch(batch_size, job, server.receiver(), |x| executor.eval_expand(&x))
            }
        }
    }
}

fn run_job_batch<X, Y>(
    batch_size: usize,
    first: Job<X, Y>,
    receiver: &Receiver<Job<X, Y>>,
    f: impl FnOnce(Vec<X>) -> Vec<Y>,
) {
    let mut all_x = vec![first.x];
    let mut senders = vec![first.sender];

    while all_x.len() < batch_size {
        let Job { x, sender } = receiver.recv().unwrap();
        all_x.push(x);
        senders.push(sender);
    }

    let all_y = f(all_x);

    for (y, sender) in zip_eq_exact(all_y, senders) {
        sender.send(y).unwrap();
    }
}
