use board_game::games::ataxx::AtaxxBoard;
use board_game::util::board_gen::random_board_with_moves;
use board_game::util::bot_game;
use rand::thread_rng;

use alpha_zero::{old_zero, zero};
use alpha_zero::mapping::ataxx::AtaxxStdMapper;
use alpha_zero::network::cudnn::CudnnNetwork;
use cuda_sys::wrapper::handle::Device;

fn main() {
    let path = "../data/ataxx/test_loop/training/gen_264/model_1_epochs.onnx";

    let random = false;
    let batch_size = 20;

    let normal_iterations = 1000;
    let batch_iterations = 10 * normal_iterations;

    let batch_exploration_weight = 2.0;
    let normal_exploration_weight = 2.0;

    if false {
        let batched_network = CudnnNetwork::load(AtaxxStdMapper, path, batch_size, Device::new(0));
        let batched_settings = zero::ZeroSettings::new(batch_size, batch_exploration_weight, random);
        let mut batched_bot = zero::ZeroBot::new(batch_iterations, batched_settings, batched_network, thread_rng());

        let old_network = CudnnNetwork::load(AtaxxStdMapper, path, 1, Device::new(0));
        let old_settings = old_zero::ZeroSettings::new(normal_exploration_weight, random);
        let mut old_bot = old_zero::ZeroBot::new(normal_iterations, old_settings, old_network, thread_rng());

        let board = AtaxxBoard::default();
        println!("{}", batched_bot.build_tree(&board).display(2));
        println!("{}", old_bot.build_tree(&board).display(2));
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    println!("{:#?}", bot_game::run(
        || random_board_with_moves(&AtaxxBoard::default(), 2, &mut thread_rng()),
        || {
            let network = CudnnNetwork::load(AtaxxStdMapper, path, batch_size, Device::new(0));
            let settings = zero::ZeroSettings::new(batch_size, batch_exploration_weight, random);
            zero::ZeroBot::new(batch_iterations, settings, network, thread_rng())
        },
        || {
            let network = CudnnNetwork::load(AtaxxStdMapper, path, 1, Device::new(0));
            let settings = old_zero::ZeroSettings::new(normal_exploration_weight, random);
            old_zero::ZeroBot::new(normal_iterations, settings, network, thread_rng())
        },
        5, true, Some(1),
    ));
}
