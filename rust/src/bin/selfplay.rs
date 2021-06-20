#![allow(unused_imports)]

use sttt::util::lower_process_priority;

use sttt_zero::network::google_torch::all_cuda_devices;
use sttt_zero::selfplay::{MCTSGenerator, MoveSelector, Settings, ZeroGenerator};

fn main() {
    lower_process_priority();

    let settings = Settings {
        position_count: 1_000,
        output_path: "../data/loop/games_0.csv".to_owned(),

        move_selector: MoveSelector {
            inf_temp_move_count: 20
        },

        generator: ZeroGenerator {
            devices: all_cuda_devices(),
            threads_per_device: 2,
            batch_size: 400,

            network_path: "../data/esat/deeper_adam/model_5_epochs.pt".to_owned(),
            iterations: 1000,
            exploration_weight: 1.0,
        },

        /*
        generator: MCTSGenerator {
            thread_count: 4,

            iterations: 1_000,
            exploration_weight: 1.0,
        },
        */
    };
    settings.run();
}
