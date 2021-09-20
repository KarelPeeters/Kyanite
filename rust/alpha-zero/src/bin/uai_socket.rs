use std::io::Write;
use std::net::TcpStream;
use std::time::Instant;

use board_game::board::Board;
use board_game::games::ataxx::AtaxxBoard;
use board_game::uai;
use rand::thread_rng;

use alpha_zero::mapping::ataxx::AtaxxStdMapper;
use alpha_zero::network::cudnn::CudnnNetwork;
use alpha_zero::zero::{zero_build_tree, ZeroSettings};
use cuda_sys::wrapper::handle::Device;

const PASSWORD: &str = "635A26AD425331A6";

fn main() -> std::io::Result<()> {
    let path = "C:/Documents/Programming/STTT/AlphaZero/data/ataxx/test_loop/training/gen_240/model_1_epochs.onnx";
    let batch_size = 100;
    let settings = ZeroSettings::new(batch_size, 2.0, true);

    let mut network = CudnnNetwork::load(AtaxxStdMapper, path, batch_size, Device::new(0));
    let mut rng = thread_rng();

    let bot = move |board: &AtaxxBoard, time_to_use| {
        let start = Instant::now();
        let stop_cond = || {
            (Instant::now() - start).as_millis() as u32 > time_to_use
        };

        let tree = zero_build_tree(board, 1_000_000_000, settings, &mut network, &mut rng, stop_cond);

        let mv = if tree[0].visits == 0 {
            board.random_available_move(&mut rng)
        } else {
            tree.best_move()
        };

        let info = format!("nodes: {}, wdl: {:?}", tree[0].visits, tree.wdl());
        (mv, info)
    };

    println!("opening socket");
    let socket = TcpStream::connect("server.ataxx.org:28028")?;

    println!("logging in");
    writeln!(&socket, "user {}", "kZero-small")?;
    writeln!(&socket, "pass {}", PASSWORD)?;

    println!("starting uai");
    uai::client::run(
        bot,
        "kZero-small",
        "KarelPeeters",
        &socket,
        &socket,
        std::io::stdout().lock(),
    )?;

    Ok(())
}
