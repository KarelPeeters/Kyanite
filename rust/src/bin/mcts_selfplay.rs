use sttt::mcts::mcts_build_tree;
use sttt::board::Board;
use sttt::mcts::heuristic::ZeroHeuristic;
use rand::{thread_rng, SeedableRng};
use std::time::Instant;
use rand::rngs::SmallRng;

fn main() {
    let start = Instant::now();
    let mut count = 0;
    let mut rng = SmallRng::from_entropy();
    loop {
        mcts_build_tree(&Board::new(), 50_000, &ZeroHeuristic, &mut rng);
        count += 1;
        let delta = (Instant::now() - start).as_secs_f64();
        println!("{:.2} moves/s", (count as f64) / delta);
    }
}