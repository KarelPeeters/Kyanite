use board_game::ai::mcts::MCTSBot;
use board_game::ai::minimax::{Heuristic, MiniMaxBot};
use board_game::ai::simple::RandomBot;
use board_game::ai::solver::{SolverBot, SolverHeuristic};
use board_game::board::Board;
use board_game::games::ataxx::AtaxxBoard;
use board_game::games::chess::ChessBoard;
use board_game::games::sttt::STTTBoard;
use board_game::games::ttt::TTTBoard;
use board_game::heuristic::ataxx::AtaxxTileHeuristic;
use board_game::heuristic::chess::ChessPieceValueHeuristic;
use board_game::heuristic::sttt::STTTTileHeuristic;
use board_game::util::game_stats::average_game_stats;
use rand::thread_rng;

fn main() {
    println!("TTT:");
    main_impl_game(&TTTBoard::default(), SolverHeuristic);
    println!("STTT:");
    main_impl_game(&STTTBoard::default(), STTTTileHeuristic::default());
    println!("Ataxx:");
    main_impl_game(&AtaxxBoard::default(), AtaxxTileHeuristic::default());
    println!("Chess:");
    main_impl_game(&ChessBoard::default(), ChessPieceValueHeuristic);
}

fn main_impl_game<B: Board>(start: &B, heuristic: impl Heuristic<B>) {
    let mut rng = thread_rng();
    let n = 100;

    println!("  Random:");
    println!("    {:?}", average_game_stats(start, RandomBot::new(&mut rng), n));
    println!("  Minimax:");
    println!("    {:?}", average_game_stats(start, MiniMaxBot::new(6, heuristic, &mut rng), n));
    println!("  Solver:");
    println!("    {:?}", average_game_stats(start, SolverBot::new(6, &mut rng), n));
    println!("  MCTS:");
    println!("    {:?}", average_game_stats(start, MCTSBot::new(100, 2.0, &mut rng), n));
}
