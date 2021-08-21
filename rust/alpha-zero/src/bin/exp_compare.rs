use cuda_nn_eval::cpu_executor::CpuExecutor;
use cuda_nn_eval::graph::Graph;
use board_game::util::bot_game;
use board_game::games::ataxx::AtaxxBoard;
use board_game::util::board_gen::random_board_with_moves;
use rand::thread_rng;
use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use cuda_sys::wrapper::handle::Device;
use alpha_zero::zero::{ZeroBot, ZeroSettings};
use rayon::ThreadBuilder;

fn main() {
    println!("{:#?}", bot_game::run(
        || random_board_with_moves(&AtaxxBoard::new_without_gaps(), 2, &mut thread_rng()),
        || {
            let network = AtaxxCNNNetwork::load("../data/ataxx/test_loop/training/gen_264/model_1_epochs.onnx", 1, Device::new(0));
            ZeroBot::new(1000, ZeroSettings::new(2.0, true), network, thread_rng())
        },
        || {
            let network = AtaxxCNNNetwork::load("../data/ataxx/test_loop/training/gen_264/model_1_epochs.onnx", 1, Device::new(0));
            ZeroBot::new(1000, ZeroSettings::new(2.0, true), network, thread_rng())
        },
        20, true, Some(1)
    ));

}