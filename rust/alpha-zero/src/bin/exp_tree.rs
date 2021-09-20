use board_game::games::ataxx::AtaxxBoard;
use board_game::util::board_gen::random_board_with_moves;
use rand::thread_rng;

use alpha_zero::mapping::ataxx::AtaxxStdMapper;
use alpha_zero::network::cudnn::CudnnNetwork;
use alpha_zero::zero::{ZeroBot, ZeroSettings};
use cuda_sys::wrapper::handle::Device;

fn main() {
    let device = Device::new(0);
    let path = "../data/ataxx/test_loop/training/gen_360/model_1_epochs.onnx";
    let iterations = 10_000;
    let batch_size = 10;
    let settings = ZeroSettings::new(batch_size, 2.0, true);
    let network = CudnnNetwork::load(AtaxxStdMapper, path, batch_size, device);
    let mut zero_bot = ZeroBot::new(iterations, settings, network, thread_rng());

    let board = random_board_with_moves(&AtaxxBoard::default(), 4, &mut thread_rng());
    println!("{}", board);

    let tree = zero_bot.build_tree(&board);
    println!("{}", tree.display(1));
    // println!("{}", tree.display(2));
}
