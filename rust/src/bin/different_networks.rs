use sttt_zero::network::Network;
use tch::Device;
use sttt::bot_game;
use sttt_zero::mcts_zero::ZeroBot;
use sttt::util::lower_process_priority;
use sttt::mcts::MCTSBot;
use rand::thread_rng;
use sttt::bot_game::Bot;
use sttt::board::{Coord, Board};
use rayon::ThreadPoolBuilder;

struct CombinedBot<A: Bot, B: Bot> {
    a: A,
    b: B,
    tile_count_threshold: u32,
}

impl<A: Bot, B: Bot> Bot for CombinedBot<A, B> {
    fn play(&mut self, board: &Board) -> Option<Coord> {
        if board.count_tiles() < self.tile_count_threshold {
            self.a.play(board)
        } else {
            self.b.play(board)
        }
    }
}

fn main() {
    lower_process_priority();

    let mcts_bot = || MCTSBot::new(100_000, thread_rng());

    ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .unwrap();

    for &threshold in &[0, 10, 20, 30, 40, 50, 60] {
        println!("Threshold {}", threshold);
        println!("{:?}", bot_game::run(
            mcts_bot,
            || CombinedBot {
                a: ZeroBot::new(1000, 1.0, Network::load("../data/esat/trained_model_10_epochs.pt", Device::Cpu)),
                b: mcts_bot(),
                tile_count_threshold: threshold,
            },
            20,
            true,
        ));
    }
}
