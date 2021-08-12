use cuda_sys::wrapper::handle::Device;
use alpha_zero::zero::{ZeroSettings, ZeroBot};
use board_game::games::ataxx::AtaxxBoard;
use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use rand::thread_rng;
use alpha_zero::bot_game_zero;
use alpha_zero::selfplay::MoveSelector;
use board_game::ai::simple::RandomBot;
use board_game::heuristic::ataxx_heuristic::AtaxxTileHeuristic;
use board_game::ai::mcts::MCTSBot;
use alpha_zero::bot_game_zero::OpponentConstructor;
use board_game::ai::minimax::MiniMaxBot;

fn main() {
    let device = Device::new(0);

    let iterations = 100;
    let settings = ZeroSettings::new(2.0, true);

    let path_main = "../data/derp/test_loop/gen_24/model_1_epochs.onnx";

    let opponents: &[OpponentConstructor<_>] = &[
        Box::new(|| Box::new(RandomBot::new(thread_rng()))),
        Box::new(|| Box::new(MiniMaxBot::new(6, AtaxxTileHeuristic::greedy(), thread_rng()))),
    ];
    bot_game_zero::run(
        opponents,
        &AtaxxBoard::new_without_gaps(),
        iterations,
        settings,
        &mut AtaxxCNNNetwork::load(path_main, opponents.len() * 10 * 2, device),
        MoveSelector::zero_temp(),
        10,
    );
}
