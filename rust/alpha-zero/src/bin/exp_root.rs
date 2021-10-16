use board_game::ai::mcts::MCTSBot;
use board_game::board::{Board, BoardAvailableMoves, Outcome};
use board_game::games::sttt::STTTBoard;
use board_game::util::board_gen::random_board_with_forced_win;
use board_game::util::bot_game;
use internal_iterator::InternalIterator;
use rand::thread_rng;

use alpha_zero::mapping::sttt::STTTStdMapper;
use alpha_zero::network::cudnn::CudnnNetwork;
use alpha_zero::network::Network;
use alpha_zero::zero::{zero_build_tree, ZeroBot, ZeroSettings};
use cuda_nn_eval::Device;
use nn_graph::onnx::load_graph_from_onnx_path;

fn main() {
    let path = "../data/newer_loop/test/training/gen_809/network.onnx";
    let batch_size = 1;
    let settings = ZeroSettings::new(batch_size, 2.0, true);
    let iterations = 10_000;
    let board = STTTBoard::default();

    let graph = load_graph_from_onnx_path(path);
    let mapper = STTTStdMapper;

    //TODO why does the zero but suck terribly?
    //TODO maybe the network arch just sucks?
    println!("{:?}", bot_game::run(
        STTTBoard::default,
        || MCTSBot::new(1000, 2.0, thread_rng()),
        || {
            let network = CudnnNetwork::new(mapper, graph.clone(), batch_size, Device::new(0));
            ZeroBot::new(100, settings, network, thread_rng())
        },
        10, true, Some(1),
    ));

    /*
    let tree = zero_build_tree(&board, iterations, settings, &mut network, &mut thread_rng(), || false);
    println!("{}", tree.display(1));
    */

    /*
    for _ in 0..100 {
        let board = random_board_with_forced_win(&STTTBoard::default(), 1, &mut thread_rng());
        println!("{}", board);

        let eval = network.evaluate(&board);
        println!("{:?}", eval.wdl);

        let mut direct_win_policy = 0.0;
        let mut direct_win_count = 0;
        let mut move_count = 0;

        board.available_moves().enumerate().for_each(|(i, mv)| {
            // TODO this is not entirely what we want, other moves could be winning too
            if board.clone_and_play(mv).outcome() == Some(Outcome::WonBy(board.next_player())) {
                direct_win_policy += eval.policy[i];
                direct_win_count += 1;
            }
            move_count += 1;
        });

        println!("Direct win policy: {}", direct_win_policy);
        println!("Uniform direct win policy: {}", direct_win_count as f32 / move_count as f32);

        println!("{}", board);
    }*/
}
