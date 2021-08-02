use std::marker::PhantomData;
use board_game::games::ataxx::AtaxxBoard;

use alpha_zero::games::ataxx_output::AtaxxBinaryOutput;
use alpha_zero::selfplay::{MoveSelector, Settings};
use alpha_zero::selfplay::generate_zero::ZeroGeneratorSettings;
use alpha_zero::zero::ZeroSettings;
use alpha_zero::games::ataxx_cnn_network::AtaxxCNNLoader;
use cuda_sys::wrapper::handle::Device;
use alpha_zero::network::tower_shape::TowerShape;

fn main() {
    let output_path = "../data/derp/derp_games.bin";
    let network_path = "../data/derp/basic_res_model/params.npz";

    let batch_size = 256;

    let settings = Settings {
        start_board: AtaxxBoard::new_without_gaps(),
        game_count: 10_000,
        output: AtaxxBinaryOutput::new(output_path),
        move_selector: MoveSelector::new(1.0,  20),
        generator: ZeroGeneratorSettings {
            batch_size,
            full_search_prob: 1.0,
            full_iterations: 800,
            part_iterations: 200,
            zero_settings: ZeroSettings::new(2.0, true),
            keep_tree: false,
            dirichlet_alpha: 0.2,
            dirichlet_eps: 0.25,
            max_game_length: 400,
            devices: Device::all().collect(),
            threads_per_device: 1,
            network: AtaxxCNNLoader {
                path: network_path.to_owned(),
                shape: TowerShape {
                    board_size: 7,
                    input_channels: 3,
                    tower_depth: 8,
                    tower_channels: 32,
                    wdl_hidden_size: 16,
                    policy_channels: 17
                },
                max_batch_size: batch_size
            },
            ph: PhantomData,
        },
    };

    settings.run();
}
