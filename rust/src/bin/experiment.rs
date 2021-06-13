use sttt_zero::network::Network;
use tch::Device;
use sttt_zero::mcts_zero::{zero_build_tree, ZeroBot};
use sttt::board::Board;
use sttt::mcts::{mcts_build_tree, MCTSBot};
use sttt::mcts::heuristic::ZeroHeuristic;
use rand::thread_rng;
use sttt::mcts;
use ordered_float::OrderedFloat;
use itertools::Itertools;
use sttt::bot_game::run;
use sttt::util::lower_process_priority;
use rayon::ThreadPoolBuilder;

fn main() {
    lower_process_priority();

    ThreadPoolBuilder::new()
        .num_threads(2)
        .build_global()
        .unwrap();

    println!("{:?}", run(
        || MCTSBot::new(100_000, thread_rng()),
        || ZeroBot::new(1000, 1.0, Network::load("../data/esat/fixed_res/trained_model_2_epochs.pt", Device::Cpu)),
        20, true
    ));

    // println!("{:?}", run(
    //     || MCTSBot::new(100_000, thread_rng()),
    //     || ZeroBot::new(1000, 1.0, Network::load("../data/esat/trained_model_10_epochs.pt", Device::Cpu)),
    //     20, true
    // ));
}
