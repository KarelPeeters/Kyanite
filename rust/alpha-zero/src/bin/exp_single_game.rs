use rand::thread_rng;

use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use board_game::board::Board;
use board_game::games::ataxx::AtaxxBoard;
use cuda_sys::wrapper::handle::Device;
use alpha_zero::zero::{ZeroSettings, ZeroBot};

fn main() {
    let path = "../data/ataxx/test_loop/training/gen_240/model_1_epochs.onnx";

    let batch_size = 20;
    let network = AtaxxCNNNetwork::load(path, batch_size, Device::new(0));

    let settings = ZeroSettings::new(batch_size, 2.0, true);
    let mut bot = ZeroBot::new(100_000, settings, network, thread_rng());

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