use rand::thread_rng;

use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use alpha_zero::zero::{zero_build_tree, ZeroBot, ZeroSettings};
use board_game::ai::minimax::{minimax, MiniMaxBot};
use board_game::board::Board;
use board_game::games::ataxx::{AtaxxBoard, Coord, Move};
use board_game::heuristic::ataxx_heuristic::AtaxxTileHeuristic;
use board_game::util::bot_game;
use cuda_sys::wrapper::handle::Device;

fn main() {
    let mut board = AtaxxBoard::from_fen("2xxo1o/xxxxoo1/xooxxxx/ooooxoo/xoooxxx/xoooooo/xox2o1 o 0 42");
    // board.play(Move::Copy { to: Coord::from_xy(3, 0) });
    println!("{}", board);

    println!("{:?}", bot_game::run(
        || board.clone(),
        || {
            let device = Device::new(0);
            let path = "../data/ataxx/test_loop/training/gen_360/model_1_epochs.onnx";
            let iterations = 100_000;
            let batch_size = 10;
            let settings = ZeroSettings::new(batch_size, 2.0, true);
            let mut network = AtaxxCNNNetwork::load(path, batch_size, device);
            let zero_bot = ZeroBot::new(iterations, settings, network, thread_rng());
            zero_bot
        },
        || MiniMaxBot::new(16, AtaxxTileHeuristic::default(), thread_rng()),
        1, true, Some(1),
    ));

    // let result = minimax(&board, &AtaxxTileHeuristic::default(), 16, &mut thread_rng());
    // println!("{:?}", result);
    // println!("{}", i32::MAX as i64 - result.value as i64);
    // println!("{}", result.value as i64 - i32::MIN as i64);
    //
    // let tree = zero_build_tree(&board, iterations, settings, &mut network, &mut thread_rng(), || false);
    // println!("{}", tree.display(2));
}
