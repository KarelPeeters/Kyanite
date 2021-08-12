use rand::thread_rng;

use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use alpha_zero::zero::{ZeroBot, ZeroSettings};
use board_game::board::Board;
use board_game::games::ataxx::AtaxxBoard;
use cuda_sys::wrapper::handle::Device;

fn main() {
    let path = "../data/derp/good_test_loop/gen_39/model_1_epochs.onnx";

    let network = AtaxxCNNNetwork::load(path, 1, Device::new(0));

    let settings = ZeroSettings::new(2.0, true);
    let mut bot = ZeroBot::new(1000, settings, network, thread_rng());

    let mut board = AtaxxBoard::from_fen("x3xox/3xxox/3xoxx/3oxxx/2ooox1/xo1oox1/xx4x");

    let tree = bot.build_tree(&board);
    println!("{}", tree.display(1));

    println!("{}", board);
    board.play(tree.best_move());
    println!("{}", board);

    println!("{}", board.to_fen());

    println!("{:?}", tree.wdl());
    println!("{:?}", tree.best_move());
}