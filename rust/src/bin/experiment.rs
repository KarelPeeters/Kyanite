use sttt_zero::mcts_zero::{mcts_zero_build_tree, mcts_zero_state_build_tree};
use sttt_zero::network::Network;
use sttt::board::Board;
use tch::Device;
use rand::thread_rng;
use std::time::Instant;

fn main() {
    sttt::util::lower_process_priority();

    let mut board = Board::new();

    let mut old_time = 0.0;
    let mut new_time = 0.0;

    loop {
        board.play(board.random_available_move(&mut thread_rng()).unwrap());
        println!("{}", board);

        if board.is_done() { break }

        let iterations = 100;
        let exploration_weight = 1.0;

        let mut network = Network::load("../data/esat/trained_model_10_epochs.pt", Device::Cpu);

        let start = Instant::now();
        let state_tree = mcts_zero_state_build_tree(&board, iterations, exploration_weight, &mut network);
        new_time += (Instant::now() - start).as_secs_f32();

        let start = Instant::now();
        let tree = mcts_zero_build_tree(&board, iterations, exploration_weight, &mut network);
        old_time += (Instant::now() - start).as_secs_f32();

        assert_eq!(tree, state_tree);
    }

    println!("{}", old_time);
    println!("{}", new_time);
}
