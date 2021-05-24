use sttt::board::Board;

use sttt_zero::mcts_zero::mcts_zero_build_tree;
use sttt_zero::network::Network;
use std::time::Instant;

fn main() {
    sttt::util::lower_process_priority();

    let iterations = 1_000;

    let mut board = Board::default();
    let mut network = Network::load("../data/esat/trained_model_10_epochs.pt");

    let start = Instant::now();
    loop {
        println!("{}", board);
        if board.is_done() { break };

        let tree = mcts_zero_build_tree(&board, iterations, 1.0, &mut network);
        board.play(tree.best_move());
    }

    let total_time = (Instant::now()- start).as_secs_f32();
    let pytorch_time = network.pytorch_time;
    let rest_time = total_time - pytorch_time;

    println!("Total: {}", total_time);
    println!("Pytorch: {:.3}, {:.2}", pytorch_time, pytorch_time / total_time);
    println!("Rest: {:.3}, {:.2}", rest_time, rest_time / total_time);

    println!("Pytorch eval throughput: {:.3}", iterations as f32 / pytorch_time);
    println!("Rest eval request throughput: {:.3}", iterations as f32 / rest_time);
}
