use board_game::util::board_gen::random_board_with_moves;
use board_game::util::bot_game;
use board_game::util::bot_game::BotGameResult;
use rand::{Rng, thread_rng};

use alpha_zero::network::cudnn::CudnnNetwork;
use alpha_zero::zero::{ZeroBot, ZeroSettings};
use cuda_sys::wrapper::handle::Device;
use alpha_zero::mapping::chess::ChessStdMapper;
use board_game::games::chess::ChessBoard;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();


    let path_l = "../data/new_loop/full_pov/training/gen_32/network.onnx";
    let path_r = "../data/new_loop/full_pov/training/gen_300/network.onnx";

    let iterations_l = 600;
    let iterations_r = 600;

    let settings_l = ZeroSettings::new(20, 2.0, true);
    let settings_r = ZeroSettings::new(20, 2.0, true);

    let result = compare_bots(
        || {
            let network = CudnnNetwork::load(ChessStdMapper, path_l, settings_l.batch_size, Device::new(0));
            ZeroBot::new(iterations_l, settings_l, network, thread_rng())
        },
        || {
            let network = CudnnNetwork::load(ChessStdMapper, path_r, settings_r.batch_size, Device::new(0));
            ZeroBot::new(iterations_r, settings_r, network, thread_rng())
        },
        true,
        true,
    ).unwrap();

}

fn compare_bots<R1: Rng, R2: Rng>(
    bot_l: impl Fn() -> ZeroBot<ChessBoard, CudnnNetwork<ChessBoard, ChessStdMapper>, R1> + Sync,
    bot_r: impl Fn() -> ZeroBot<ChessBoard, CudnnNetwork<ChessBoard, ChessStdMapper>, R2> + Sync,
    tree: bool,
    game: bool,
) -> Option<BotGameResult> {
    if tree {
        let board = ChessBoard::default();
        println!("{}", bot_l().build_tree(&board).display(2));
        println!("{}", bot_r().build_tree(&board).display(2));
    }

    if game {
        let result = bot_game::run(
            || random_board_with_moves(&ChessBoard::default(), 2, &mut thread_rng()),
            bot_l,
            bot_r,
            20, true, Some(1),
        );
        println!("{:#?}", result);
        Some(result)
    } else {
        None
    }
}
