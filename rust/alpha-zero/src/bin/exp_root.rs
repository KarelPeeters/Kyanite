use board_game::games::ttt::TTTBoard;
use board_game::util::board_gen::random_board_with_forced_win;
use rand::thread_rng;

use alpha_zero::mapping::ttt::TTTStdMapper;
use alpha_zero::network::cudnn::CudnnNetwork;
use alpha_zero::zero::wrapper::ZeroSettings;
use cuda_nn_eval::Device;
use nn_graph::onnx::load_graph_from_onnx_path;

fn main() {
    let path = "../data/newer_loop/test-diri/ttt/training/gen_20/network.onnx";

    let batch_size = 10;
    let settings = ZeroSettings::new(batch_size, 2.0);
    let iterations = 1000;
    let board = random_board_with_forced_win(&TTTBoard::default(), 3, &mut thread_rng());

    let graph = load_graph_from_onnx_path(path);
    let mapper = TTTStdMapper;

    let mut network = CudnnNetwork::new(mapper, graph, batch_size, Device::new(0));

    let tree = settings.build_tree(&board, &mut network, &iterations);
    println!("{}", tree.display(1, true));
}
