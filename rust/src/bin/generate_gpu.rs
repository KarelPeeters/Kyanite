use sttt_zero::network::Network;
use tch::Device;
use sttt_zero::mcts_zero::{ZeroState, Tree, RunResult, Response};
use sttt::board::Board;
use itertools::Itertools;
use std::time::Instant;
use itertools::zip_eq;
use sttt::util::lower_process_priority;
use crossbeam::channel::Receiver;
use crossbeam::channel::Sender;
use std::process::exit;

fn main() {
    lower_process_priority();

    let thread_count = 8;

    let (start_send, start_rec) = crossbeam::channel::bounded(thread_count);
    let (count_send, count_rec) = crossbeam::channel::bounded(thread_count);


    for _ in 0..thread_count {
        let start_rec = start_rec.clone();
        let count_send = count_send.clone();
        std::thread::spawn(|| {
            runner(start_rec, count_send);
        });
    }

    let mut total_eval_count = 0;
    let mut total_move_count = 0;

    let start = Instant::now();

    println!("Sending start");
    for _ in 0..thread_count {
        start_send.send(()).unwrap();
    }

    let mut i = 0;
    for (eval_count, move_count) in count_rec {
        i += 1;

        total_eval_count += eval_count;
        total_move_count += move_count;

        if i % 100 == 0 {
            let elapsed = (Instant::now() - start).as_secs_f32();
            println!("Eval throughput: {:.2} boards/s", (total_eval_count as f32) / elapsed);
            println!("Game throughput: {:.2} moves/s", (total_move_count as f32) / elapsed);
        }

        if total_move_count > 500 {
            println!("Move threshold reached.");
            exit(0);
        }
    }

    // let pytorch_time = network.pytorch_time;
    // let other_time = elapsed - pytorch_time;
    // println!("Pytorch time fraction: {:.2}", pytorch_time / elapsed);
    // println!("Other time fraction: {:.2}", other_time / elapsed);

    fn runner(start_rec: Receiver<()>, count_send: Sender<(usize, usize)>) {
        let batch_size = 100;
        let iterations = 800;
        let exploration_weight = 1.0;

        let new_state = || ZeroState::new(Tree::new(Board::new()), iterations, exploration_weight);

        let mut network = Network::load("../data/esat/trained_model_10_epochs.pt", Device::Cuda(0));

        let mut states = (0..batch_size).map(|_| new_state()).collect_vec();
        let mut responses = (0..batch_size).map(|_| None).collect_vec();
        let mut requests = Vec::new();

        start_rec.recv().unwrap();
        println!("Starting");

        //TODO actually start playing some games instead of just computing the empty board
        loop {
            let mut move_count = 0;

            //collect requests
            for i in 0..batch_size {
                let state = &mut states[i];

                //keep running until we get an evaluation request
                loop {
                    let result = state.run(responses[i].take());
                    match result {
                        RunResult::Request(request) => {
                            requests.push(request);
                            break;
                        }
                        RunResult::Done => {
                            move_count += 1;
                            *state = new_state();
                        }
                    }
                }
            }

            //pass through network and construct responses
            let boards = requests.iter().map(|r| r.board.clone()).collect_vec();
            let mut evaluations = network.evaluate_all(&boards);

            //construct responses
            for (i, (request, evaluation)) in zip_eq(requests.drain(..), evaluations.drain(..)).enumerate() {
                assert!(responses[i].is_none());
                responses[i] = Some(Response { request, evaluation });
            }

            count_send.send((batch_size, move_count)).unwrap();
        }
    }
}