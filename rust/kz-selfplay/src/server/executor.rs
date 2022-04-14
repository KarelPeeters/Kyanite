use crate::server::job_channel::{Job, JobServer};
use board_game::board::Board;
use crossbeam::channel::{Receiver, TryRecvError};
use crossbeam::select;
use cuda_sys::wrapper::handle::Device;
use kz_core::mapping::BoardMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::muzero::{EvalResponsePair, ExpandArgs, MuZeroFusedGraphs};
use kz_core::network::{Network, ZeroEvaluation};
use kz_util::zip_eq_exact;
use nn_graph::graph::Graph;
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

pub fn executor_loop_alphazero<B: Board, M: BoardMapper<B>>(
    device: Device,
    batch_size: usize,
    mapper: M,
    graph_receiver: Receiver<Arc<Graph>>,
    server: JobServer<Vec<B>, Vec<ZeroEvaluation>>,
) {
    // wait for the initial graph
    let graph = graph_receiver.recv().unwrap();
    let mut network = CudaNetwork::new(mapper, &graph, batch_size, device);

    // handle incoming messages
    loop {
        match receiver_executor_message(&graph_receiver, &server) {
            ExecutorMessage::Graph(graph) => {
                drop(network);
                network = CudaNetwork::new(mapper, &graph, batch_size, device);
            }
            ExecutorMessage::Job(job) => {
                job.run(|x| network.evaluate_batch(&x));
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
    let mut executor = graph.root_executor(device, batch_size);

    // handle incoming messages
    loop {
        match receiver_executor_message(&graph_receiver, &server) {
            ExecutorMessage::Graph(graph) => {
                drop(executor);
                executor = graph.root_executor(device, batch_size);
            }
            ExecutorMessage::Job(Job { x, sender }) => {
                let mut all_x = vec![x];
                let mut senders = vec![sender];

                // collect additional jobs (non-blocking) to try to fill the batch size
                while all_x.len() < batch_size {
                    match server.receiver().try_recv() {
                        Ok(Job { x, sender }) => {
                            all_x.push(x);
                            senders.push(sender);
                        }
                        Err(TryRecvError::Empty) => {
                            println!("No more root eval requests available");
                            break;
                        }
                        Err(TryRecvError::Disconnected) => panic!("Root executor channel connection closed"),
                    }
                }

                println!("Calling root executor with batch size {}", all_x.len());
                let all_y = executor.eval_root(&all_x);

                for (y, sender) in zip_eq_exact(all_y, senders) {
                    sender.send(y).unwrap();
                }
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
