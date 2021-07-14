use std::ffi::OsStr;
use std::fmt::Write;
use std::path::PathBuf;

use itertools::Itertools;
use serde::Deserialize;

use sttt_zero::network::google_torch::all_cuda_devices;
use sttt_zero::selfplay::{MoveSelector, Settings};
use sttt_zero::selfplay::generate_zero::settings_torch::GoogleTorchSettings;
use sttt_zero::selfplay::generate_zero::ZeroGeneratorSettings;
use sttt_zero::zero::ZeroSettings;

#[derive(Debug, Deserialize)]
struct Args {
    output_path: String,
    network_path: String,

    game_count: u64,

    // move selection
    inf_temp_move_count: u32,
    keep_tree: bool,
    dirichlet_alpha: f32,
    dirichlet_eps: f32,

    // zero search
    full_search_prob: f64,
    full_iterations: u64,
    part_iterations: u64,
    exploration_weight: f32,
    random_symmetries: bool,

    // performance
    batch_size: usize,
    threads_per_device: usize,
}

fn main() {
    let args = std::env::args().collect_vec();
    assert_eq!(2, args.len(), "expected one argument");

    let args: Args = serde_json::from_str(&args[1])
        .expect("Failed to parse json argument");

    let output_path = PathBuf::from(&args.output_path);
    assert_eq!(Some(OsStr::new("csv")), output_path.extension());
    let settings_output_path = output_path.with_extension("txt");

    let settings = Settings {
        game_count: args.game_count,
        output_path: args.output_path,
        move_selector: MoveSelector { zero_temp_move_count: args.inf_temp_move_count },
        generator: ZeroGeneratorSettings {
            batch_size: args.batch_size,
            full_search_prob: args.full_search_prob,
            full_iterations: args.full_iterations,
            part_iterations: args.part_iterations,
            zero_settings: ZeroSettings {
                exploration_weight: args.exploration_weight,
                random_symmetries: args.random_symmetries,
            },
            keep_tree: args.keep_tree,
            dirichlet_alpha: args.dirichlet_alpha,
            dirichlet_eps: args.dirichlet_eps,
            network: GoogleTorchSettings {
                path: args.network_path,
                devices: all_cuda_devices(),
                threads_per_device: args.threads_per_device,
            },
        },
    };

    let mut settings_str = String::new();
    write!(&mut settings_str, "{:#?}", settings).unwrap();

    println!("Running with settings:\n{}", settings_str);
    std::fs::write(settings_output_path, settings_str)
        .expect("Failed to write settings file");

    settings.run();
}
