use board_game::games::chess::ChessBoard;
use board_game::games::ttt::TTTBoard;
use board_game::util::board_gen::random_board_with_forced_win;
use rand::thread_rng;

use alpha_zero::mapping::chess::ChessStdMapper;
use alpha_zero::mapping::ttt::TTTStdMapper;
use alpha_zero::network::cudnn::CudnnNetwork;
use alpha_zero::zero::wrapper::ZeroSettings;
use cuda_nn_eval::Device;
use nn_graph::onnx::load_graph_from_onnx_path;

fn main() {
    let path = "../data/supervised/lichess_huge/network_1028.onnx";

    let batch_size = 100;
    let settings = ZeroSettings::new(batch_size, 2.0);
    let iterations = 100_000;
    let board = ChessBoard::default();

    println!("{}", board);

    let graph = load_graph_from_onnx_path(path);
    let mapper = ChessStdMapper;

    let mut network = CudnnNetwork::new(mapper, graph, batch_size, Device::new(0));

    let tree = settings.build_tree(&board, &mut network, &iterations);
    println!("{}", tree.display(1, true));
}
