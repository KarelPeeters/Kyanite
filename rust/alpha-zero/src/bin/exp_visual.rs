use std::str::FromStr;

use board_game::ai::Bot;
use board_game::board::Board;
use board_game::games::chess::{ChessBoard, moves_to_pgn, Rules};
use board_game::games::max_length::MaxMovesBoard;

use alpha_zero::mapping::chess::ChessStdMapper;
use alpha_zero::network::cpu::CPUNetwork;
use alpha_zero::network::cudnn::CudnnNetwork;
use alpha_zero::stats::visualize::visualize_network_activations;
use alpha_zero::zero::wrapper::{ZeroBot, ZeroSettings};
use cuda_nn_eval::Device;
use nn_graph::onnx::load_graph_from_onnx_path;

fn main() {
    let graph = load_graph_from_onnx_path("../data/newer_loop/easy_start10/chess/training/gen_36/network.onnx");
    println!("{}", graph);

    let mapper = ChessStdMapper;

    let batch_size = 10;
    let network = CudnnNetwork::new(mapper, graph.clone(), batch_size, Device::new(0));
    let settings = ZeroSettings::new(batch_size, 2.0);
    let mut bot = ZeroBot::new(network, settings, 600);

    let mut boards = vec![];
    let mut moves = vec![];

    let inner = chess::Board::from_str("1r1k2r1/8/8/8/8/8/8/R2K3Q w - - 0 1").unwrap();
    let inner = ChessBoard::new(inner, Rules::default());
    let mut board = MaxMovesBoard::new(inner, 50);

    while !board.is_done() {
        println!("{:?}", board);
        boards.push(board.inner().clone());
        let mv = bot.select_move(board.inner());
        moves.push(mv);
        board.play(mv);
    }

    println!("{}", moves_to_pgn(&moves));

    let mut network = CPUNetwork::new(mapper, graph.clone());
    let images = visualize_network_activations(&mut network, &boards);

    let _ = std::fs::remove_dir_all("ignored/visualize");
    std::fs::create_dir_all("ignored/visualize").unwrap();
    for (i, image) in images.iter().enumerate() {
        image.save(format!("ignored/visualize/board_{}.png", i)).unwrap();
    }
}