#![allow(unused_imports)]

use sttt::util::lower_process_priority;

use sttt_zero::mcts_zero::ZeroSettings;
use sttt_zero::network::google_torch::all_cuda_devices;
use sttt_zero::selfplay::{MoveSelector, Settings};
use sttt_zero::selfplay::generate_mcts::MCTSGeneratorSettings;
use sttt_zero::selfplay::generate_zero::settings_torch::GoogleTorchSettings;
use sttt_zero::selfplay::generate_zero::ZeroGeneratorSettings;

fn main() {
    lower_process_priority();

    let settings = Settings {
        position_count: 200_000,
        output_path: "../data/loop2/games.csv".to_owned(),

        move_selector: MoveSelector {
            inf_temp_move_count: 20
        },

        generator: ZeroGeneratorSettings {
            batch_size: 500,

            iterations: 5_000,
            zero_settings: ZeroSettings::new(2.0, true),

            keep_tree: false,
            dirichlet_alpha: 1.0,
            dirichlet_eps: 0.25,

            network: GoogleTorchSettings {
                path: "../data/loop/modest_cont22/model_15_epochs.pt".to_owned(),
                devices: all_cuda_devices(),
                threads_per_device: 2,
            },
        },
    };
    settings.run();
}
