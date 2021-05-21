use rand::thread_rng;
use sttt::board::Board;
use sttt::mcts::heuristic::ZeroHeuristic;
use sttt::mcts::{mcts_build_tree, MCTSBot};

use sttt_zero::mcts_zero::mcts_zero_build_tree;
use sttt_zero::network::Network;
use sttt::bot_game;
use sttt::bots::RandomBot;
use sttt::bot_game::Bot;

fn main() {
    sttt::util::lower_process_priority();

    let c = 1.0;
    let game_count = 1;
    let iterations_zero = 5_000;
    let iterations_mcts = 100_000;

    let network_path = "../data/esat/trained_model_10_epochs.pt";
    let mut network = Network::load(network_path);

    println!("zero({}, c={}) playing against itself", iterations_zero, c);
    println!("  using network {}", network_path);
    println!("positions also evaluated by mcts({})", iterations_mcts);
    print!("\n\n\n");

    for _ in 0..game_count {
        let mut i = 0;
        let mut board = Board::default();

        while !board.is_done() {
            let zero_tree = mcts_zero_build_tree(&board, iterations_zero, c, &mut network);
            let mcts_tree = mcts_build_tree(&board, iterations_mcts, &ZeroHeuristic, &mut thread_rng());

            print!("{}", board);

            println!("next player: {:?}", board.next_player);

            //signs here chosen to show advantage for first player
            println!(
                "value: mcts({:.4}), zero({:.4}), net({:.4})",
                -zero_tree[0].value(),
                -mcts_tree[0].signed_value(),
                zero_tree[0].evaluation.unwrap(),
            );

            println!("moves: (mcts P, mcts V), (zero P, zero V), (net P, net V)");
            let mcts_root_visits = mcts_tree[0].visits as f32;
            let zero_root_visits = zero_tree[0].visits as f32;

            for (i, mv) in board.available_moves().enumerate() {
                let mcts_child = &mcts_tree[mcts_tree[0].children().unwrap().get(i)];
                let zero_child = &zero_tree[zero_tree[0].children().unwrap().get(i)];

                //signs here to print things from the perspective of the previous player
                println!(
                    "  {:?}: mcts({:.4}, {:.4}), zero({:.4}, {:.4}), net({:.4}, {:.4})",
                    mv,
                    mcts_child.visits as f32 / mcts_root_visits,
                    mcts_child.signed_value(),
                    zero_child.visits as f32 / zero_root_visits,
                    zero_child.value(),
                    zero_child.policy,
                    -zero_child.evaluation.unwrap_or(f32::NAN),
                );
            }

            //TODO try playing with noise like in real data generation
            let mv = zero_tree.best_move();
            i = i + 1;
            board.play(mv);


            print!("\n\n\n");
        }

        print!("{}", board);
        println!("won by: {:?}", board.won_by);
    }
}
