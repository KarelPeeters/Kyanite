use std::time::Instant;

use rand::thread_rng;

use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use alpha_zero::zero::{zero_build_tree, ZeroSettings};
use board_game::uai;
use cuda_sys::wrapper::handle::Device;
use board_game::board::Board;
use board_game::games::ataxx::AtaxxBoard;

fn main() -> std::io::Result<()> {
    let path = "C:/Documents/Programming/STTT/AlphaZero/data/ataxx/test_loop/training/gen_240/model_1_epochs.onnx";
    let batch_size = 20;
    let settings = ZeroSettings::new(batch_size, 2.0, true);

    let mut network = AtaxxCNNNetwork::load(path, batch_size, Device::new(0));
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

    uai::client::run(
        bot,
        "kZero",
        "Karel Peeters",
        std::io::stdin().lock(),
        std::io::stdout().lock(),
        std::fs::File::create("ignored/log.txt")?,
    )?;

    Ok(())
}