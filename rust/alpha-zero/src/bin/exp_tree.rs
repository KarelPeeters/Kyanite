use rand::thread_rng;

use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use alpha_zero::zero::{zero_build_tree, ZeroSettings};
use board_game::games::ataxx::AtaxxBoard;
use cuda_sys::wrapper::handle::Device;
use board_game::util::board_gen::{random_board_with_forced_win, random_board_with_moves};
use board_game::board::Board;
use board_game::ai::minimax::minimax;
use board_game::heuristic::ataxx_heuristic::AtaxxTileHeuristic;

fn main() {
    let device = Device::new(0);

    let mut rng = thread_rng();

    let iterations = 100_000;
    let settings = ZeroSettings::new(2.0, true);

    let paths = [
        "../data/derp/test_loop/gen_240/model_1_epochs.onnx",
    ];

    for _ in 0..10 {
        let board = random_board_with_moves(&AtaxxBoard::new_without_gaps(), 20, &mut rng);
        if board.is_done() { continue; }


        println!("{}", board);
        // board.play(Move::Jump { from: Coord::from_xy(1, 6), to: Coord::from_xy(3, 4) });
        // println!("{}", board);

        for &path in &paths {
            println!("{}:", path);
            let mut network = AtaxxCNNNetwork::load(path, 1, device);
            let tree = zero_build_tree(&board, iterations, settings, &mut network, &mut rng, || false);
            println!("{}", tree.display(1));
        }

        println!("{:?}", minimax(&board, &AtaxxTileHeuristic::default(), 6, &mut rng));
    }
}
