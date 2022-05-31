use std::cmp::min;
use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};

use board_game::board::Board;
use flume::{Receiver, RecvError, Selector, Sender, TryRecvError};
use futures::never::Never;
use itertools::Itertools;
use superluminal_perf::{begin_event_with_color, end_event};

use cuda_sys::wrapper::handle::Device;
use kz_core::mapping::BoardMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::job_channel::{Job, JobServer};
use kz_core::network::{Network, ZeroEvaluation};
use nn_graph::graph::Graph;

use crate::superluminal::{CL_BLUE, CL_GREEN, CL_YELLOW};

#[derive(Debug, Copy, Clone)]
pub enum RunCondition {
    FullBatch,
    JobCount(usize),
    Any,
}

pub fn batched_executor_loop<G, N, X, Y>(
    max_batch_size: usize,
    run_condition: RunCondition,
    graph_receiver: Receiver<Option<G>>,
    server: JobServer<X, Y>,
    load_network: impl Fn(G) -> N,
    evaluate_batch: impl Fn(&mut N, &[X]) -> Vec<Y>,
) {
    let thread_name = std::thread::current().name().unwrap_or("unnamed").to_owned();
    assert_ne!(max_batch_size, 0, "Got batch size 0 for {}", thread_name);

    let job_receiver = server.into_receiver();

    let mut state = State::new();
    let mut network: Option<N> = None;

    // this is a separate flag and not just `graph_receiver.is_disconnected`
    //   to ensure we have properly handled the disconnection event itself
    let mut graph_disconnected = false;

    loop {
        // check that there is at least one channel to wait on
        assert!(network.is_some() || !graph_disconnected);
        let mut selector = Selector::new();

        // only wait for graphs if the channel is still open
        if !graph_disconnected {
            selector = selector.recv(&graph_receiver, Message::Graph);
        }

        // only wait for jobs if we already have a network, otherwise we're just pointlessly filling our buffers
        if network.is_some() {
            selector = selector.recv(&job_receiver, Message::Job);
        }

        // block until we get a new message
        begin_event_with_color("wait", CL_YELLOW);
        let message = selector.wait();
        end_event();

        let _: Never = match message {
            Message::Graph(Ok(graph)) => {
                handle_new_graph(&mut network, graph, &load_network, &thread_name);
                continue;
            }
            Message::Job(job) => {
                // safe because we only listen to the job channel if we have a network
                let network = network.as_mut().unwrap();

                match job {
                    Ok(job) => {
                        // we've got a new job, great
                        state.push_job(job);

                        // check for additional jobs (non-blocking)
                        //   we could loop again but we might as well just fall back to the outer loop
                        while state.x.len() < max_batch_size {
                            match job_receiver.try_recv() {
                                // yay, we've got an additional job
                                Ok(job) => state.push_job(job),
                                // we've run out of non-blocking jobs
                                Err(TryRecvError::Empty) => break,
                                // let the outer match deal with it
                                Err(TryRecvError::Disconnected) => continue,
                            }
                        }

                        // optionally evaluate some batches
                        if state.should_eval(run_condition, max_batch_size) {
                            run_eval(&mut state, network, &evaluate_batch, max_batch_size);
                        }

                        continue;
                    }
                    Err(RecvError::Disconnected) => {
                        // the job channel has disconnected

                        // evaluate all remaining jobs if any
                        assert!(state.items_to_eval() < max_batch_size);
                        if state.items_to_eval() > 0 {
                            run_eval(&mut state, network, &evaluate_batch, max_batch_size);
                        }
                        assert!(state.items_to_eval() == 0 && state.items_to_send() == 0);

                        // at this point we can safely exit, we won't get any more jobs
                        //  this also closes the graph receiver
                        return;
                    }
                }
            }
            Message::Graph(Err(RecvError::Disconnected)) => {
                // we'll never get a new graph again
                match &network {
                    Some(_) => {
                        // .. but we do have a final network to continue evaluating batches with
                        graph_disconnected = true;
                        continue;
                    }
                    None => {
                        // ... and we don't even have a network, make sure that that's okay
                        // leftover jobs?
                        if state.items_to_eval() > 0 {
                            panic!(
                                "Executor {}: graph disconnected by we still have items pending: {:?}",
                                thread_name, state
                            );
                        }
                        // wait for job receiver disconnection
                        match job_receiver.recv() {
                            Ok(_) => panic!("Executor {}: got new job after graph disconnection", thread_name),
                            Err(RecvError::Disconnected) => {}
                        }
                        // we can safely exit this executor now that both channels are disconnected
                        return;
                    }
                }
            }
        };
    }
}

pub fn alphazero_batched_executor_loop<B: Board, M: BoardMapper<B>>(
    max_batch_size: usize,
    device: Device,
    mapper: M,
    run_condition: RunCondition,
    graph: Graph,
    server: JobServer<B, ZeroEvaluation<'static>>,
) {
    let (graph_sender, graph_receiver) = flume::bounded(1);
    graph_sender.send(Some(graph)).unwrap();
    drop(graph_sender);

    batched_executor_loop(
        max_batch_size,
        run_condition,
        graph_receiver,
        server,
        |graph| CudaNetwork::new(mapper, &graph, max_batch_size, device),
        |network, batch_x| network.evaluate_batch(&batch_x),
    )
}

#[derive(Debug)]
enum Message<G, J> {
    Graph(G),
    Job(J),
}

struct State<X, Y> {
    x: VecDeque<X>,
    senders: VecDeque<(usize, Sender<Vec<Y>>)>,
    leftover_y: VecDeque<Y>,
}

impl<X, Y> Debug for State<X, Y> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State")
            .field("items_to_eval", &self.items_to_eval())
            .field("items_waiting_for_send", &self.items_waiting_for_send())
            .field("items_to_send", &self.items_to_send())
            .field("senders", &self.senders.len())
            .finish()
    }
}

impl<X, Y> State<X, Y> {
    pub fn new() -> Self {
        Self {
            x: VecDeque::new(),
            senders: VecDeque::new(),
            leftover_y: VecDeque::new(),
        }
    }

    fn check_invariants(&self) {
        assert!(!self.can_fill_next_sender());
        assert_eq!(
            self.items_to_eval() + self.items_waiting_for_send(),
            self.items_to_send()
        );
    }

    fn items_to_eval(&self) -> usize {
        self.x.len()
    }

    fn items_waiting_for_send(&self) -> usize {
        self.leftover_y.len()
    }

    fn items_to_send(&self) -> usize {
        self.senders.iter().map(|&(len, _)| len).sum::<usize>()
    }

    fn push_job(&mut self, job: Job<X, Y>) {
        // TODO we could avoid an extra vec copy if all batches we receive "happen to" have exact size max_batch_size
        //   by keeping the single leading batch as a separate vec and falling back to deque if necessary
        //   this is pretty complicated and shouldn't matter too much though

        let Job { x, sender } = job;

        if x.len() == 0 {
            // avoid ever putting empty senders in the queue since that introduces tricky edge cases
            let _ = sender.send(vec![]);
        } else {
            self.senders.push_back((x.len(), sender));
            self.x.extend(x.into_iter());
        }

        self.check_invariants();
    }

    fn should_eval(&self, cond: RunCondition, max_batch_size: usize) -> bool {
        if self.x.len() == 0 {
            return false;
        }
        if self.x.len() >= max_batch_size {
            return true;
        }

        match cond {
            RunCondition::FullBatch => false,
            RunCondition::JobCount(count) => self.senders.len() >= count,
            RunCondition::Any => self.x.len() > 0,
        }
    }

    fn get_batch(&mut self, max_batch_size: usize) -> &[X] {
        let batch_size = min(self.x.len(), max_batch_size);
        assert_ne!(batch_size, 0);

        if self.x.as_slices().0.len() < batch_size {
            self.x.make_contiguous();
        }

        &self.x.as_slices().0[0..batch_size]
    }

    fn can_fill_next_sender(&self) -> bool {
        match self.senders.get(0) {
            None => {
                assert_eq!(self.leftover_y.len(), 0);
                false
            }
            Some(&(count, _)) => self.leftover_y.len() >= count,
        }
    }

    // Distribute the given response over the appropriate senders.
    // We ignore SendError here, since it's not our fault that the receiver stopped caring about the response.
    fn respond_batch(&mut self, batch_y: Vec<Y>) {
        let batch_size = batch_y.len();

        // remove just-evaluated x-values
        assert!(batch_size <= self.x.len());
        drop(self.x.drain(0..batch_size));

        if self.leftover_y.is_empty() && self.senders[0].0 == batch_size {
            // shortcut to avoid extra copies, just send the entire vec immediately
            let _ = self.senders.pop_front().unwrap().1.send(batch_y);
        } else {
            // add results to temporary storage
            self.leftover_y.extend(batch_y.into_iter());

            // send as many values out as possible
            while self.can_fill_next_sender() {
                let (count, sender) = self.senders.pop_front().unwrap();
                let block_y = self.leftover_y.drain(0..count).collect_vec();
                let _ = sender.send(block_y);
            }
        }

        self.check_invariants();
    }
}

fn run_eval<X, Y, N>(
    state: &mut State<X, Y>,
    network: &mut N,
    evaluate_batch: &impl Fn(&mut N, &[X]) -> Vec<Y>,
    max_batch_size: usize,
) {
    begin_event_with_color("run", CL_GREEN);
    let batch_x = state.get_batch(max_batch_size);
    let batch_y = evaluate_batch(network, batch_x);
    end_event();

    begin_event_with_color("reply", CL_YELLOW);
    state.respond_batch(batch_y);
    end_event();
}

fn handle_new_graph<N, G>(network: &mut Option<N>, graph: Option<G>, load_network: impl Fn(G) -> N, thread_name: &str) {
    // drop previous network if any to save GPU memory
    if let Some(network) = network.take() {
        begin_event_with_color("drop", CL_BLUE);
        println!("{} dropping network", thread_name);
        drop(network);
        end_event();
    }

    // load the new network if any
    *network = graph.map(|graph| {
        begin_event_with_color("load", CL_BLUE);
        println!("{} loading new network", thread_name);
        let network = load_network(graph);
        end_event();
        network
    });
}
