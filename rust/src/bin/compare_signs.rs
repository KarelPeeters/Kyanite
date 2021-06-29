use rand::rngs::SmallRng;
use rand::SeedableRng;
use sttt::board::board_to_compact_string;
use sttt::board_gen::random_board_with_forced_win;
use sttt::mcts::mcts_build_tree;
use sttt::minimax::evaluate_minimax;

use sttt_zero::mcts_zero::{zero_build_tree, ZeroSettings};
use sttt_zero::network::dummy::DummyNetwork;
use sttt_zero::network::google_onnx::GoogleOnnxNetwork;

fn main() {
    let mut rng = SmallRng::from_entropy();

    let win_depth = 9;
    println!("Generated board with forced win in {}:", win_depth);
    let board = random_board_with_forced_win(win_depth, &mut rng);
    // let board = Board::new();

    println!("{}", board);
    println!("{:?}", board_to_compact_string(&board));

    println!("\nMinimax:");
    let mm_eval = evaluate_minimax(&board, 11);
    println!("{:?}", mm_eval);

    println!("\nMCTS:");
    let mcts_tree = mcts_build_tree(&board, 1_000_000, 2.0, &mut rng);
    println!("value: {}", mcts_tree.eval().value());
    println!("best move: {:?}", mcts_tree.best_move());
    mcts_tree.print(1);

    let zero_settings = ZeroSettings::new(2.0, true);

    println!("\nZero:");
    let mut network = GoogleOnnxNetwork::load("../data/esat/modest/model_4_epochs.onnx");
    let zero_tree = zero_build_tree(&board, 100_000, zero_settings, &mut network, &mut rng);
    println!("value: {}", zero_tree.value());
    println!("best move: {:?}", zero_tree.best_move());
    println!("{}", zero_tree.display(1));

    println!("\nZero Empty:");
    let dummy_tree = zero_build_tree(&board, 100_000, zero_settings, &mut DummyNetwork, &mut rng);
    println!("value: {}", dummy_tree.value());
    println!("best move: {:?}", dummy_tree.best_move());
    println!("{}", dummy_tree.display(1));
}
