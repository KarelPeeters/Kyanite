#![allow(unused_imports)]

use sttt::util::lower_process_priority;

use sttt_zero::network::google_torch::{all_cuda_devices, GoogleTorchNetwork};
use sttt_zero::selfplay::{GoogleOnnxSettings, MCTSGeneratorSettings, MoveSelector, Settings, ZeroGeneratorSettings};

fn main() {
    lower_process_priority();

    let settings = Settings {
        position_count: 1_000,
        output_path: "../data/loop/games_0.csv".to_owned(),

        move_selector: MoveSelector {
            inf_temp_move_count: 20
        },

        generator: ZeroGeneratorSettings {
            /*
            network: GoogleTorchNetworkSettings {
                devices: all_cuda_devices(),
                threads_per_device: 2,
                path: "../data/esat/modest/resave_2_epochs.onnx".to_owned(),
            },
            */

            //TODO figure out why onnx is not running on the GPU
            network: GoogleOnnxSettings {
                // path: "../data/esat/modest/resave_2_epochs.onnx".to_owned(),
                path: "../data/onnx/small.onnx".to_owned(),
                num_threads: 1,
            },

            batch_size: 1000,
            iterations: 1000,
            exploration_weight: 1.0,
        },

        /*
        generator: MCTSGeneratorSettings {
            thread_count: 4,

            iterations: 1_000,
            exploration_weight: 1.0,
        },
        */
    };
    settings.run();
}
