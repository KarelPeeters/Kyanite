use rand::thread_rng;

use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use alpha_zero::games::ataxx_torch_network::AtaxxTorchNetwork;
use alpha_zero::network::Network;
use alpha_zero::util::PanicRng;
use alpha_zero::zero::{Tree, zero_build_tree, ZeroBot, ZeroSettings};
use board_game::games::ataxx::AtaxxBoard;
use board_game::util::bot_game;
use cuda_sys::wrapper::handle::Device;
use std::time::Instant;

fn main() {
    let torch_path = "../data/derp/basic_res_model/model.pt";
    let onnx_path = "../data/derp/basic_res_model/model.onnx";

    let mut torch_network = AtaxxTorchNetwork::load(torch_path, tch::Device::Cuda(0));
    let mut cnn_network = AtaxxCNNNetwork::load(onnx_path, 1, Device::new(0));

    println!("Root board eval");
    let board = AtaxxBoard::new_without_gaps();
    println!("{}", board);

    println!("{:?}", torch_network.evaluate(&board));
    println!("{:?}", cnn_network.evaluate(&board));

    println!("Tree");
    fn tree(network: &mut impl Network<AtaxxBoard>) -> Tree<AtaxxBoard> {
        let board = AtaxxBoard::new_without_gaps();
        let start = Instant::now();
        let tree = zero_build_tree(
            &board,
            1000, ZeroSettings::new(2.0, false),
            network,
            &mut PanicRng,
        );
        println!("Took {}s", (Instant::now() - start).as_secs_f32());
        tree
    }

    println!("Torch:");
    println!("{}", tree(&mut torch_network).display(4));

    println!("CNN:");
    println!("{}", tree(&mut cnn_network).display(4));

    println!("bot_game");
    let settings = ZeroSettings::new(2.0, true);
    println!("{:#?}", bot_game::run(
        || AtaxxBoard::new_without_gaps(),
        || {
            let network = AtaxxTorchNetwork::load(torch_path, tch::Device::Cuda(0));
            ZeroBot::new(100, settings, network, thread_rng())
        },
        || {
            let network = AtaxxCNNNetwork::load(onnx_path, 1, Device::new(0));
            ZeroBot::new(100, settings, network, thread_rng())
        },
        1, true, Some(1),
    ));
}