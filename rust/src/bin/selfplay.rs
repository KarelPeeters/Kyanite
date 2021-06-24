#![allow(unused_imports)]

use sttt::util::lower_process_priority;

use sttt_zero::selfplay::{MoveSelector, Settings};
use sttt_zero::selfplay::generate_mcts::MCTSGeneratorSettings;
use sttt_zero::selfplay::generate_zero::ZeroGeneratorSettings;

fn main() {
    lower_process_priority();

    let settings = Settings {
        position_count: 1_000,
        output_path: "../data/loop/games_0.csv".to_owned(),

        move_selector: MoveSelector {
            inf_temp_move_count: 20
        },

        generator: MCTSGeneratorSettings {
            thread_count: 4,

            iterations: 100_000,
            exploration_weight: 1.0,
        },
    };
    settings.run();
}
