use board_game::ai::minimax::minimax;
use board_game::ai::solver::{is_double_forced_draw, solve_all_moves, SolverHeuristic};
use board_game::board::{Board, BoardAvailableMoves, Outcome};
use board_game::games::ttt::TTTBoard;
use board_game::util::game_stats::all_possible_boards;
use board_game::wdl::POV;
use internal_iterator::InternalIterator;
use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use alpha_zero::mapping::ttt::TTTStdMapper;
use alpha_zero::network::cudnn::CudnnNetwork;
use alpha_zero::network::ZeroEvaluation;
use alpha_zero::stats::network_accuracy::{Challenge, network_accuracy};
use cuda_nn_eval::Device;
use nn_graph::onnx::load_graph_from_onnx_path;

fn main() {
    let depth = 15;
    let start = TTTBoard::default();
    let boards = all_possible_boards(&start, false);

    let challenges = boards.into_iter().map(|board| {
        assert!(!board.is_done());

        is_double_forced_draw(&board, depth).expect("Must be solvable");

        let eval = solve_all_moves(&board, depth);

        let available_move_count = board.available_moves().count();
        let best_moves = eval.best_move.unwrap();
        let is_optimal = board.available_moves().map(|mv| best_moves.contains(&mv)).collect();

        let policy = board.available_moves().map(|mv| {
            if best_moves.contains(&mv) {
                1.0 / best_moves.len() as f32
            } else {
                0.0
            }
        }).collect();

        let outcome = eval.value.to_outcome_wdl().unwrap().un_pov(board.next_player());

        let solution = ZeroEvaluation {
            wdl: outcome.pov(board.next_player()).to_wdl(),
            policy,
        };
        Challenge { board, solution, is_optimal: Some(is_optimal) }
    }).collect_vec();

    let device = Device::new(0);
    let graph = load_graph_from_onnx_path("../data/supervised/all-ttt/network_3328.onnx");
    let mut network = CudnnNetwork::new(TTTStdMapper, graph, 1, device);

    network_accuracy(&mut network, &challenges[0..100])
}