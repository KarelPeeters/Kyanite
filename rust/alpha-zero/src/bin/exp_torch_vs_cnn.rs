use alpha_zero::games::ataxx_torch_network::AtaxxTorchNetwork;
use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use cuda_sys::wrapper::handle::Device;
use alpha_zero::zero::{zero_build_tree, Tree, ZeroSettings, ZeroBot};
use board_game::games::ataxx::AtaxxBoard;
use alpha_zero::network::Network;
use alpha_zero::util::PanicRng;
use board_game::util::bot_game;
use rand::thread_rng;
use alpha_zero::network::tower_shape::TowerShape;

fn main() {
    let torch_path = "../data/derp/basic_res_model/training/model_1_epochs.pt";
    let cnn_path = "../data/derp/basic_res_model/params.npz";

    let shape = TowerShape {
        board_size: 7,
        input_channels: 3,
        tower_depth: 8,
        tower_channels: 32,
        wdl_hidden_size: 16,
        policy_channels: 17
    };

    let mut torch_network = AtaxxTorchNetwork::load(torch_path, tch::Device::Cuda(0));
    let mut cnn_network = AtaxxCNNNetwork::load(cnn_path, &shape.to_graph(1), 1, Device::new(0));

    let settings = ZeroSettings::new(2.0, true);
    println!("{:#?}", bot_game::run(
        || AtaxxBoard::new_without_gaps(),
        || {
            let network = AtaxxTorchNetwork::load(torch_path, tch::Device::Cuda(0));
            ZeroBot::new(100, settings, network, thread_rng())
        },
        || {
            let network = AtaxxCNNNetwork::load(cnn_path, &shape.to_graph(1), 1, Device::new(0));
            ZeroBot::new(100, settings, network, thread_rng())
        },
        1, true, Some(1),
    ));

    fn tree(network: &mut impl Network<AtaxxBoard>) -> Tree<AtaxxBoard> {
        let board = AtaxxBoard::new_without_gaps();
        zero_build_tree(
            &board,
            1000, ZeroSettings::new(2.0, false),
            network,
            &mut PanicRng,
        )
    }

    println!("Torch:");
    println!("{}", tree(&mut torch_network).display(4));

    println!("CNN:");
    println!("{}", tree(&mut cnn_network).display(4));
}