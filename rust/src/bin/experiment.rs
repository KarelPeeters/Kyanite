use sttt_zero::network::Network;
use tch::Device;
use sttt_zero::mcts_zero::{zero_build_tree, ZeroBot};
use sttt::board::{Board, Coord};
use sttt::mcts::{mcts_build_tree, MCTSBot};
use sttt::mcts::heuristic::ZeroHeuristic;
use rand::thread_rng;
use sttt::mcts;
use ordered_float::OrderedFloat;
use itertools::Itertools;
use sttt::bot_game::run;
use sttt::util::lower_process_priority;
use rayon::ThreadPoolBuilder;
use sttt_zero::network::google_torch::GoogleTorchNetwork;

fn main() {
    lower_process_priority();

    let mut network = GoogleTorchNetwork::load("../data/esat/deeper_adam/model_5_epochs.pt", Device::Cpu);

    let mut board = Board::new();
    println!("{}", board);
    println!("{}", zero_build_tree(&board, 10_000, 1.0, &mut network).display(4));

    board.play(Coord::from_oo(4, 4));
    println!("{}", board);
    println!("{}", zero_build_tree(&board, 10_000, 1.0, &mut network).display(4));

    board.play(Coord::from_oo(4, 0));
    println!("{}", board);
    println!("{}", zero_build_tree(&board, 10_000, 1.0, &mut network).display(4));
}
