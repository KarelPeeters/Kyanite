use rand::thread_rng;

use alpha_zero::bot_game_zero;
use alpha_zero::bot_game_zero::OpponentConstructor;
use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use alpha_zero::selfplay::core::MoveSelector;
use alpha_zero::old_zero::{ZeroBot, ZeroSettings};
use board_game::games::ataxx::AtaxxBoard;
use board_game::util::board_gen::random_board_with_moves;
use cuda_sys::wrapper::handle::Device;

fn main() {
    let device = Device::new(0);

    let iterations = 100;
    let settings = ZeroSettings::new(2.0, true);

    let path_main = "../data/ataxx/test_loop/training/gen_264/model_1_epochs.onnx";
    let path_other = "../data/ataxx/test_loop/training/gen_249/model_1_epochs.onnx";

    let opponents: &[OpponentConstructor<_>] = &[
        Box::new(move || Box::new(ZeroBot::new(iterations, settings, AtaxxCNNNetwork::load(&path_other, 1, device), thread_rng()))),
    ];

    let start_board = || {
        let mut rng = thread_rng();
        random_board_with_moves(&AtaxxBoard::default(), 2, &mut rng)
    };

    bot_game_zero::run(
        opponents,
        start_board,
        iterations,
        settings,
        &mut AtaxxCNNNetwork::load(path_main, opponents.len() * 10 * 2, device),
        MoveSelector::zero_temp(),
        10,
    );
}
