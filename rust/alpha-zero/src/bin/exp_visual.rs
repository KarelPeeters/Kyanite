use std::str::FromStr;

use board_game::games::chess::{ChessBoard, Rules};
use board_game::util::game_stats::all_possible_boards;
use itertools::Itertools;

use alpha_zero::mapping::chess::ChessStdMapper;
use alpha_zero::network::cpu::CPUNetwork;
use alpha_zero::stats::visualize::visualize_network_activations_split;
use nn_graph::onnx::load_graph_from_onnx_path;

fn main() {
    let path = std::fs::read_to_string("ignored/network_path.txt").unwrap();
    let graph = load_graph_from_onnx_path(path);
    println!("{}", graph);

    let mapper = ChessStdMapper;

    // let batch_size = 10;
    // let network = CudnnNetwork::new(mapper, graph.clone(), batch_size, Device::new(0));
    // let settings = ZeroSettings::new(batch_size, 2.0);

    // let fen = "1r1k2r1/8/8/8/8/8/8/R2K3Q w - - 0 1";

    let board = ChessBoard::default();

    let mut boards = all_possible_boards(&board, 2, false);
    println!("board count: {}", boards.len());

    let pgns = [
        "k7/ppp5/P7/8/8/4R3/5PPP/7K b - - 0 1",
        "k7/ppp5/P7/8/8/4R3/3q1PPP/7K b - - 0 1",
    ];
    let mut extra_boards = pgns.iter()
        .map(|pgn| ChessBoard::new(chess::Board::from_str(pgn).unwrap(), Rules::default()))
        .collect_vec();

    extra_boards.append(&mut boards);

    let mut network = CPUNetwork::new(mapper, graph.clone());
    let (images_a, images_b) = visualize_network_activations_split(&mut network, &extra_boards, None);

    std::fs::create_dir_all("ignored/visualize").unwrap();
    for (i, image) in images_a.iter().enumerate() {
        image.save(format!("ignored/visualize/board_a_{}.png", i)).unwrap();
    }
    for (i, image) in images_b.iter().enumerate() {
        image.save(format!("ignored/visualize/board_b_{}.png", i)).unwrap();
    }
}