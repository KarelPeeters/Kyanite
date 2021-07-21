use board_game::ai::minimax::MiniMaxBot;
use board_game::heuristic::ataxx_heuristic::AtaxxTileHeuristic;
use board_game::uai;
use rand::thread_rng;

fn main() -> std::io::Result<()> {
    uai::client::run(MiniMaxBot::new(4, AtaxxTileHeuristic::default(), thread_rng()))
}