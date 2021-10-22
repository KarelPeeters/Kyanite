use std::str::FromStr;

use board_game::games::chess::{ChessBoard, Rules};
use board_game::games::max_length::MaxMovesBoard;
use board_game::util::game_stats::all_possible_boards;
use itertools::Itertools;

use alpha_zero::mapping::chess::ChessStdMapper;
use alpha_zero::network::cpu::CPUNetwork;
use alpha_zero::stats::visualize::visualize_network_activations;
use nn_graph::onnx::load_graph_from_onnx_path;

fn main() {
    let graph = load_graph_from_onnx_path("../data/newer_loop/easy_start10/chess/training/gen_36/network.onnx");
    println!("{}", graph);

    let mapper = ChessStdMapper;

    // let batch_size = 10;
    // let network = CudnnNetwork::new(mapper, graph.clone(), batch_size, Device::new(0));
    // let settings = ZeroSettings::new(batch_size, 2.0);

    let inner = chess::Board::from_str("1r1k2r1/8/8/8/8/8/8/R2K3Q w - - 0 1").unwrap();
    let inner = ChessBoard::new(inner, Rules::default());
    let board = MaxMovesBoard::new(inner, 50);

    let boards = all_possible_boards(&board, 2, false);
    let boards = boards.iter().map(|b| b.inner()).collect_vec();
    println!("board count: {}", boards.len());

    let mut network = CPUNetwork::new(mapper, graph.clone());
    let (images_a, images_b) = visualize_network_activations(&mut network, &boards);

    std::fs::create_dir_all("ignored/visualize").unwrap();
    for (i, image) in images_a.iter().enumerate() {
        image.save(format!("ignored/visualize/board_a_{}.png", i)).unwrap();
    }
    for (i, image) in images_b.iter().enumerate() {
        image.save(format!("ignored/visualize/board_b_{}.png", i)).unwrap();
    }
}