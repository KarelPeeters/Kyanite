use sttt::ai::minimax::MiniMaxBot;
use sttt::heuristic::ataxx_heuristic::AtaxxTileHeuristic;
use sttt::uai;

fn main() -> std::io::Result<()> {
    uai::client::run(MiniMaxBot::new(4, AtaxxTileHeuristic::default()))
}