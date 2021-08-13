use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use cuda_sys::wrapper::handle::Device;
use alpha_zero::zero::{ZeroSettings, ZeroBot, zero_build_tree};
use board_game::uai;
use rand::thread_rng;
use board_game::ai::Bot;
use std::time::Instant;

fn main() {
    let path = "C:/Documents/Programming/STTT/AlphaZero/data/derp/good_test_loop/gen_40/model_1_epochs.onnx";
    let settings = ZeroSettings::new(2.0, true);

    let mut network = AtaxxCNNNetwork::load(path, 1, Device::new(0));
    let mut rng = thread_rng();

    uai::client::run(move |board, time_left| {
        let start = Instant::now();
        let stop_cond = || {
            (Instant::now() - start).as_millis() as u32 > time_left * 7 / 10
        };

        let tree= zero_build_tree(board, 1_000_000, settings, &mut network, &mut rng, stop_cond);
        (tree.best_move(), tree[0].visits)
    } ).unwrap();
}