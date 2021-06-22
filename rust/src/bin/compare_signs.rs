use rand::SeedableRng;
use rand::rngs::SmallRng;
use sttt::board::board_to_compact_string;
use sttt::board_gen::random_board_with_forced_win;
use sttt::mcts::mcts_build_tree;
use sttt::minimax::evaluate_minimax;
use tch::Device;

use sttt_zero::mcts_zero::zero_build_tree;
use sttt_zero::network::google_torch::GoogleTorchNetwork;

fn main() {
    let mut rng = SmallRng::seed_from_u64(0);
    let board = random_board_with_forced_win(3, &mut rng);

    println!("{}", board);
    println!("{}", board_to_compact_string(&board));

    println!("\nMinimax:");
    let mm_eval = evaluate_minimax(&board, 10);
    println!("{:?}", mm_eval);

    println!("\nMCTS:");
    let mcts_tree = mcts_build_tree(&board, 100_000, 2.0, &mut rng);
    mcts_tree.print(1);
    println!("value: {}", mcts_tree.eval().value());

    println!("\nZero:");
    let mut network = GoogleTorchNetwork::load("../data/esat/deeper_adam/model_5_epochs.pt", Device::Cpu);

    let zero_tree = zero_build_tree(&board, 5000, 1.0, &mut network);
    println!("{}", zero_tree.display(1));
    println!("value: {}", zero_tree.value());
}
