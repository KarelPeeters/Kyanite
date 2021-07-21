use std::ffi::OsStr;
use std::fmt::Write;
use std::marker::PhantomData;
use std::path::PathBuf;

use itertools::Itertools;
use serde::Deserialize;
use sttt::games::ataxx::AtaxxBoard;

use sttt_zero::games::ataxx_output::AtaxxBinaryOutput;
use sttt_zero::games::ataxx_torch_network::AtaxxTorchSettings;
use sttt_zero::network::torch_utils::all_cuda_devices;
use sttt_zero::selfplay::{MoveSelector, Settings};
use sttt_zero::selfplay::generate_zero::ZeroGeneratorSettings;
use sttt_zero::zero::ZeroSettings;

#[derive(Debug, Deserialize)]
struct Args {
    game: String,
    game_count: u64,

    output_path: String,
    network_path: String,

    // move selection
    temperature: f32,
    zero_temp_move_count: u32,

    keep_tree: bool,
    dirichlet_alpha: f32,
    dirichlet_eps: f32,

    max_game_length: u32,

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

    assert_eq!("ataxx", args.game, "Only ataxx implemented for now");

    let output_path = PathBuf::from(&args.output_path);
    assert_eq!(Some(OsStr::new("bin")), output_path.extension());
    let settings_output_path = output_path.with_extension("txt");

    let settings = Settings {
        start_board: AtaxxBoard::new_without_gaps(),
        game_count: args.game_count,
        output: AtaxxBinaryOutput::new(args.output_path),
        move_selector: MoveSelector::new(args.temperature, args.zero_temp_move_count),
        generator: ZeroGeneratorSettings {
            batch_size: args.batch_size,
            full_search_prob: args.full_search_prob,
            full_iterations: args.full_iterations,
            part_iterations: args.part_iterations,
            zero_settings: ZeroSettings::new(args.exploration_weight, args.random_symmetries),
            keep_tree: args.keep_tree,
            dirichlet_alpha: args.dirichlet_alpha,
            dirichlet_eps: args.dirichlet_eps,
            max_game_length: args.max_game_length,
            network: AtaxxTorchSettings {
                path: args.network_path,
                devices: all_cuda_devices(),
                threads_per_device: args.threads_per_device,
            },
            ph: PhantomData,
        },
    };

    let mut settings_str = String::new();
    write!(&mut settings_str, "{:#?}", settings).unwrap();

    println!("Running with settings:\n{}", settings_str);
    std::fs::write(settings_output_path, settings_str)
        .expect("Failed to write settings file");

    settings.run();
}
