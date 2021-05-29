use sttt::board::Board;

use sttt_zero::mcts_zero::{mcts_zero_build_tree, mcts_zero_expand_tree, Tree};
use sttt_zero::network::Network;

fn main() {
    sttt::util::lower_process_priority();

    let iterations = 1000;

    println!("Without tree keeping:");
    let mut network = Network::load("../data/esat/trained_model_10_epochs.pt");
    let mut board = Board::new();

    while !board.is_done() {
        let tree = mcts_zero_build_tree(&board, iterations, 1.0, &mut network);
        board.play(tree.best_move());
    }

    println!("{} eval requests, {} distinct", network.total_eval_count, network.cache.len());

    println!("With tree keeping:");
    let mut network = Network::load("../data/esat/trained_model_10_epochs.pt");
    let mut board = Board::new();

    let mut prev_tree: Option<Tree> = None;

    while !board.is_done() {

        let tree = match prev_tree.take() {
            Some(mut tree) => {
                let iterations_left = iterations - tree[0].visits;
                mcts_zero_expand_tree(&mut tree, iterations_left, 1.0, &mut network);
                tree
            }
            None => mcts_zero_build_tree(&board, iterations, 1.0, &mut network),
        };

        board.play(tree.best_move());
        prev_tree = Some(tree);
    }

    println!("{} eval requests, {} distinct", network.total_eval_count, network.cache.len());
}
