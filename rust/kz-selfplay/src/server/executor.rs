use flume::{Receiver, Selector};

use kz_util::zip_eq_exact;

use crate::server::job_channel::{Job, JobServer};
use crate::server::server::GraphReceiver;

#[derive(Debug)]
enum ExecutorMessage<G, X, Y> {
    Graph(G),
    Job(Job<X, Y>),
}

pub fn batched_executor_loop<G, N, X, Y>(
    batch_size: usize,
    graph_receiver: GraphReceiver<G>,
    server: JobServer<X, Y>,
    load_network: impl Fn(&G) -> N,
    evaluate_batch: impl Fn(&mut N, Vec<X>) -> Vec<Y>,
) {
    let thread_name = std::thread::current().name().unwrap().to_owned();
    let mut network: Option<N> = None;

    loop {
        let message = if network.is_some() {
            // wait for either a graph or a request
            receive_executor_message(&graph_receiver, &server)
        } else {
            // block until we get a graph
            ExecutorMessage::Graph(graph_receiver.recv().unwrap())
        };

        match message {
            ExecutorMessage::Graph(graph) => {
                if network.is_some() {
                    println!("{} dropping network", thread_name);
                }

                drop(network);

                network = graph.map(|graph| {
                    println!("{} loading new network", thread_name);
                    load_network(&*graph)
                })
            }
            ExecutorMessage::Job(job) => {
                let network = network.as_mut().unwrap();

                // continue collecting requests until we fill a batch
                run_job_batch(batch_size, job, server.receiver(), |x| evaluate_batch(network, x));
            }
        }
    }
}

fn receive_executor_message<G, X, Y>(
    graph_receiver: &Receiver<G>,
    server: &JobServer<X, Y>,
) -> ExecutorMessage<G, X, Y> {
    Selector::new()
        .recv(graph_receiver, |graph| ExecutorMessage::Graph(graph.unwrap()))
        .recv(server.receiver(), |job| ExecutorMessage::Job(job.unwrap()))
        .wait()
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
