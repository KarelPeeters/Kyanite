use rand::thread_rng;
use sttt::bot_game;
use sttt::mcts::MCTSBot;

use sttt_zero::mcts_zero::MCTSZeroBot;
use sttt_zero::network::Network;

fn main() {
    //TODO try to profile this program, see if all time is spent inside of libtorch
    //  and think about batching and even gpu processing
    sttt::util::lower_process_priority();

    for i in 1..10 {
        let c = (i as f32) / 10.0;

        println!("zero-10e-1k(c={:.2}) vs mcts-100k", c);
        println!("{:?}", bot_game::run(
            || MCTSZeroBot::new(1_000, c, Network::load("../data/esat/trained_model_10_epochs.pt")),
            || MCTSBot::new(100_000, thread_rng()),
            20, true,
        ));
    }
}
