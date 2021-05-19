use rand::thread_rng;
use sttt::bot_game;
use sttt::bots::RandomBot;

use sttt_zero::mcts_zero::MCTSZeroBot;
use sttt_zero::network::Network;
use rayon::ThreadPoolBuilder;
use sttt::mcts::MCTSBot;

fn main() {
    //TODO try to profile this program, see if all time is spent inside of libtorch
    //  and think about batching and even gpu processing
    sttt::util::lower_process_priority();

    ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .expect("Failed to set global thread pool");

    println!("trained vs 50k");
    println!("{:?}", bot_game::run(
        || MCTSZeroBot::new(1_000, Network::load("../data/esat/trained_model.pt")),
        || MCTSBot::new(50_000, thread_rng()),
        20, true,
    ));
}
