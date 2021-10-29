use board_game::board::Board;
use board_game::games::chess::{ChessBoard, moves_to_pgn};

use alpha_zero::mapping::chess::ChessStdMapper;
use alpha_zero::network::cudnn::CudnnNetwork;
use alpha_zero::zero::wrapper::{ZeroBot, ZeroSettings};
use cuda_nn_eval::Device;
use nn_graph::onnx::load_graph_from_onnx_path;

fn main() {
    let path = "C:/Documents/Programming/STTT/AlphaZero/data/supervised/lichess_huge/network_5140.onnx";
    let batch_size = 100;
    let settings = ZeroSettings::new(batch_size, 2.0);

    let graph = load_graph_from_onnx_path(path);
    let network = CudnnNetwork::new(ChessStdMapper, graph, batch_size, Device::new(0));

    let mut bot = ZeroBot::new(network, settings, 10_000);

    let mut board = ChessBoard::default();
    let mut moves = vec![];

    while !board.is_done() {
        let tree = bot.build_tree(&board);
        println!("{}", tree.display(1, true));
        let mv = tree.best_move();
        moves.push(mv);
        board.play(mv);
        println!("{}", moves_to_pgn(&moves));
        println!("{}", board);
    }
}