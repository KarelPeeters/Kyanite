use rand::thread_rng;
use rayon::ThreadPoolBuilder;
use sttt::bot_game;
use sttt::mcts::MCTSBot;
use sttt::util::lower_process_priority;

use sttt_zero::mcts_zero::ZeroBot;
use sttt_zero::network::google_onnx::GoogleOnnxNetwork;

fn main() {
    lower_process_priority();

    ThreadPoolBuilder::new()
        .num_threads(2)
        .build_global().unwrap();

    println!("{:?}", bot_game::run(
        || MCTSBot::new(100_000, 2.0, thread_rng()),
        || ZeroBot::new(1_000, 2.0, GoogleOnnxNetwork::load("../data/esat/modest/model_4_epochs.onnx")),
        20, true, Some(1),
    ));
}
