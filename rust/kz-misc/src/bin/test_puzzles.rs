use std::fs::{read_to_string, File};
use std::io::BufReader;

use board_game::board::{Board, BoardMoves, Outcome};
use board_game::chess::ChessMove;
use board_game::games::chess::{ChessBoard, Rules};
use decorum::Total;
use internal_iterator::InternalIterator;
use itertools::Itertools;
use rand::rngs::StdRng;
use rand::SeedableRng;

use cuda_nn_eval::Device;
use kz_core::mapping::chess::ChessStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::wrapper::ZeroSettings;
use kz_misc::eval::lichess_puzzle::for_each_lichess_puzzle;
use nn_graph::onnx::load_graph_from_onnx_path;
use nn_graph::optimizer::optimize_graph;

fn main() {
    let path = read_to_string("ignored/network_path.txt").unwrap();

    let graph = optimize_graph(&load_graph_from_onnx_path(path), Default::default());

    let settings = ZeroSettings::simple(100, UctWeights::default(), QMode::wdl(), FpuMode::Relative(0.0));
    let visits = 10_000;

    let mapper = ChessStdMapper;
    let mut network = CudaNetwork::new(mapper, &graph, settings.batch_size, Device::new(0));
    let mut rng = StdRng::from_entropy();

    let puzzle_path = "../data/lichess/lichess_db_puzzle.csv";
    let puzzle_read = BufReader::new(File::open(puzzle_path).unwrap());

    for_each_lichess_puzzle(puzzle_read, |puzzle| {
        println!("Trying puzzle {:?}", puzzle);

        let mut moves = puzzle.moves.split(" ");

        // set up the board
        let mut board = ChessBoard::new_without_history_fen(puzzle.fen, Rules::default());
        board.play(board.parse_move(moves.next().unwrap()).unwrap());

        let player = board.next_player();
        let won_outcome = Some(Outcome::WonBy(player));

        for correct_mv in moves {
            let correct_mv = board.parse_move(correct_mv).unwrap();
            let correct_next_board = board.clone_and_play(correct_mv);
            let is_mate = correct_next_board.outcome() == won_outcome;

            if board.next_player() == player {
                let correct_moves: Vec<_> = board
                    .available_moves()
                    .filter(|&mv| is_correct_move(&board, is_mate, correct_mv, mv))
                    .collect();
                println!(
                    "Correct moves: {}",
                    correct_moves.iter().map(|mv| mv.to_string()).join(", ")
                );

                let mut zero_correct_policy_history = vec![];

                // see if we can find the move
                let tree = settings.build_tree(&board, &mut network, &mut rng, |tree| {
                    if tree.root_visits() > 0 {
                        let zero_correct_policy: f32 = tree[0]
                            .children
                            .unwrap()
                            .iter()
                            .map(|c| {
                                if correct_moves.contains(&tree[c].last_move.unwrap()) {
                                    tree[c].complete_visits as f32 / (tree[0].complete_visits - 1) as f32
                                } else {
                                    0.0
                                }
                            })
                            .sum();
                        zero_correct_policy_history.push(zero_correct_policy);
                    }

                    tree.root_visits() >= visits
                });

                let net_best_child = tree[0]
                    .children
                    .unwrap()
                    .iter()
                    .max_by_key(|&c| Total::from(tree[c].net_policy))
                    .unwrap();
                let net_mv = tree[net_best_child].last_move.unwrap();
                let net_mv_policy = tree[net_best_child].net_policy;

                let zero_mv = tree.best_move().unwrap();
                let zero_best_child = tree[0]
                    .children
                    .unwrap()
                    .iter()
                    .find(|&c| tree[c].last_move == Some(zero_mv))
                    .unwrap();
                let zero_mv_policy =
                    tree[zero_best_child].complete_visits as f32 / (tree[0].complete_visits - 1) as f32;

                let correct_children = tree[0]
                    .children
                    .unwrap()
                    .iter()
                    .filter(|&c| correct_moves.contains(&tree[c].last_move.unwrap()))
                    .collect_vec();

                let net_correct_policy: f32 = correct_children.iter().map(|&c| tree[c].net_policy).sum();
                let zero_correct_policy: f32 = correct_children
                    .iter()
                    .map(|&c| tree[c].complete_visits as f32 / (tree[0].complete_visits - 1) as f32)
                    .sum();

                let net_is_correct = is_correct_move(&board, is_mate, correct_mv, net_mv);
                let zero_is_correct = is_correct_move(&board, is_mate, correct_mv, zero_mv);

                println!("Net:");
                println!("  eval           {}", tree[0].net_values.unwrap());
                println!("  best move      {}, {:.4}, {}", net_mv, net_mv_policy, net_is_correct);
                println!("  correct policy {:.4}", net_correct_policy);

                println!("Zero:");
                println!("  eval           {}", tree[0].values());
                println!(
                    "  best move      {}, {:.4}, {}",
                    zero_mv, zero_mv_policy, zero_is_correct
                );
                println!("  correct policy {:.4}", zero_correct_policy);
                println!("    history      {:.4?}", zero_correct_policy_history);

                if !zero_is_correct {
                    println!();
                    println!("{}", tree.display(2, true, usize::MAX, false));
                }
            }

            // "play" the correct move
            board = correct_next_board;
            println!();
        }

        println!();
        println!();
    })
}

fn is_correct_move(board: &ChessBoard, is_mate: bool, correct_mv: ChessMove, mv: ChessMove) -> bool {
    let mate_outcome = Outcome::WonBy(board.next_player());
    mv == correct_mv || (is_mate && board.clone_and_play(mv).outcome() == Some(mate_outcome))
}
