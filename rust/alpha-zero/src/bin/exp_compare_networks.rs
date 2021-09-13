use board_game::games::ataxx::AtaxxBoard;
use board_game::util::board_gen::random_board_with_moves;
use board_game::util::bot_game;
use board_game::util::bot_game::BotGameResult;
use rand::{Rng, thread_rng};

use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use alpha_zero::zero::{ZeroBot, ZeroSettings};
use cuda_sys::wrapper::handle::Device;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    let mut wdls = vec![];

    for gen in [0, 10, 20, 50, 100, 150, 200, 250, 300, 350, 378] {
        println!("Comparing gen {}", gen);

        let path_l = &format!("../data/ataxx/test_loop/training/gen_{}/model_1_epochs.onnx", gen);
        let path_r = &format!("../data/derp/retrain_other/training/gen_{}/model_1_epochs.onnx", gen);

        let iterations_l = 1000;
        let iterations_r = 1000;

        let settings_l = ZeroSettings::new(20, 2.0, true);
        let settings_r = ZeroSettings::new(20, 2.0, true);

        let result = compare_bots(
            || {
                let network = AtaxxCNNNetwork::load(path_l, settings_l.batch_size, Device::new(0));
                ZeroBot::new(iterations_l, settings_l, network, thread_rng())
            },
            || {
                let network = AtaxxCNNNetwork::load(path_r, settings_r.batch_size, Device::new(0));
                ZeroBot::new(iterations_r, settings_r, network, thread_rng())
            },
            false,
            true,
        ).unwrap();

        wdls.push(vec![result.win_rate_l, result.draw_rate, result.win_rate_r, result.elo_l]);
    }

    println!("{:?}", wdls);
}

fn compare_bots<R1: Rng, R2: Rng>(
    bot_l: impl Fn() -> ZeroBot<AtaxxBoard, AtaxxCNNNetwork, R1> + Sync,
    bot_r: impl Fn() -> ZeroBot<AtaxxBoard, AtaxxCNNNetwork, R2> + Sync,
    tree: bool,
    game: bool,
) -> Option<BotGameResult> {
    if tree {
        let board = AtaxxBoard::default();
        println!("{}", bot_l().build_tree(&board).display(2));
        println!("{}", bot_r().build_tree(&board).display(2));
    }

    if game {
        let result = bot_game::run(
            || random_board_with_moves(&AtaxxBoard::default(), 2, &mut thread_rng()),
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
