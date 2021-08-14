use std::time::Instant;

use rand::thread_rng;

use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use alpha_zero::zero::{zero_build_tree, ZeroSettings};
use board_game::uai;
use cuda_sys::wrapper::handle::Device;
use board_game::board::Board;

fn main() {
    let path = "C:/Documents/Programming/STTT/AlphaZero/data/derp/test_loop/gen_240/model_1_epochs.onnx";
    let settings = ZeroSettings::new(2.0, true);

    let mut network = AtaxxCNNNetwork::load(path, 1, Device::new(0));
    let mut rng = thread_rng();

    uai::client::run(move |board, time_left| {
        let start = Instant::now();
        let stop_cond = || {
            (Instant::now() - start).as_millis() as u32 > time_left / 30
        };

        let tree = zero_build_tree(board, 1_000_000, settings, &mut network, &mut rng, stop_cond);

        let mv = if tree[0].visits == 0 {
            board.random_available_move(&mut rng)
        } else {
            tree.best_move()
        };

        (mv, tree[0].visits)
    }).unwrap();
}