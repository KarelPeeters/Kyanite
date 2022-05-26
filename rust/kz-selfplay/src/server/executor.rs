use flume::{Receiver, Selector};
use superluminal_perf::{begin_event_with_color, end_event};

use kz_util::sequence::zip_eq_exact;

use crate::server::job_channel::{Job, JobServer};
use crate::server::server::GraphReceiver;
use crate::superluminal::{CL_BLUE, CL_GREEN, CL_YELLOW};

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
    evaluate_batch: impl Fn(&mut N, &[X]) -> Vec<Y>,
) {
    let thread_name = std::thread::current().name().unwrap().to_owned();
    assert_ne!(batch_size, 0, "Got batch size 0 for {}", thread_name);

    let mut network: Option<N> = None;

    let mut batch_x = vec![];
    let mut batch_senders = vec![];

    loop {
        // Wait for the next message. This is interleaved with filling a batch so we can never deadlock if graphs are coming in quickly.
        begin_event_with_color("wait", CL_YELLOW);
        let message = if network.is_some() && batch_x.len() < batch_size {
            receive_executor_message(&graph_receiver, &server)
        } else {
            // the batch is full already, we need a graph
            ExecutorMessage::Graph(graph_receiver.recv().unwrap())
        };
        end_event();

        match message {
            ExecutorMessage::Graph(graph) => {
                // drop network before loading new one to save some GPU memory
                if network.is_some() {
                    begin_event_with_color("drop", CL_BLUE);
                    println!("{} dropping network", thread_name);
                    drop(network);
                    end_event();
                }

                // load the new network if any
                network = graph.map(|graph| {
                    begin_event_with_color("load", CL_BLUE);
                    println!("{} loading new network", thread_name);
                    let network = load_network(&*graph);
                    end_event();
                    network
                });
            }
            ExecutorMessage::Job(Job { x, sender }) => {
                batch_x.push(x);
                batch_senders.push(sender);
            }
        }

        // if we have both a network and a full batch, evaluate it
        if let Some(network) = &mut network {
            if batch_x.len() == batch_size {
                begin_event_with_color("run", CL_GREEN);
                let batch_y = evaluate_batch(network, &batch_x);
                end_event();

                begin_event_with_color("reply", CL_YELLOW);
                for (y, sender) in zip_eq_exact(batch_y, &batch_senders) {
                    sender.send(y).unwrap();
                }
                end_event();

                batch_x.clear();
                batch_senders.clear();
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
