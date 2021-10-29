use std::str::FromStr;

use board_game::board::Board;
use board_game::games::chess::{ChessBoard, Rules};

use alpha_zero::mapping::chess::ChessStdMapper;
use alpha_zero::network::cudnn::CudnnNetwork;
use alpha_zero::zero::wrapper::ZeroSettings;
use cuda_nn_eval::Device;
use nn_graph::onnx::load_graph_from_onnx_path;

fn main() {
    let path = "../data/supervised/lichess_huge_lr/network_5140.onnx";

    let batch_size = 1;
    let settings = ZeroSettings::new(batch_size, 2.0);
    let iterations = 2;

    let graph = load_graph_from_onnx_path(path);
    let mapper = ChessStdMapper;

    let mut network = CudnnNetwork::new(mapper, graph, batch_size, Device::new(0));

    let fens = [
        ("7Q/8/8/8/2k5/5N2/PPP2PK1/8 b - - 0 1", vec![]),
        ("7Q/8/8/8/2k5/5N2/PPP2PK1/8 b - - 0 1", vec!["c4b4"]),
        // "7Q/8/8/8/2k5/5N2/PPP2PK1/8 w - - 0 1",
    ];

    for (fen, moves) in fens {
        let mut board = ChessBoard::new(chess::Board::from_str(fen).unwrap(), Rules::default());
        for mv in moves {
            board.play(board.parse_move(mv).unwrap())
        }

        println!("{}", board);

        let _tree = settings.build_tree(&board, &mut network, |tree| tree.root_visits() >= iterations);
        // println!("{}", tree.display(10, true));

        println!();
        println!();
    }
}
