use itertools::Itertools;

use alpha_zero::games::ataxx_cnn_network::AtaxxCNNNetwork;
use alpha_zero::games::ataxx_output::AtaxxBinaryOutput;
use alpha_zero::new_selfplay::core::{Command, Settings, StartupSettings};
use alpha_zero::new_selfplay::server::selfplay_server_main;
use board_game::games::ataxx::AtaxxBoard;
use cuda_sys::wrapper::handle::Device;

fn main() {
    print_example_commands();

    // let startup_settings = parse_startup();
    let startup_settings = StartupSettings {
        game: "ataxx".to_string(),
        output_folder: "../data/derp/selfplay/".to_string(),
        threads_per_device: 2,
        batch_size: 256,
        games_per_file: 1000,

        tower_shape: TowerShape {
            board_size: 7,
            input_channels: 3,
            tower_depth: 8,
            tower_channels: 32,
            wdl_hidden_size: 16,
            policy_channels: 17,
        }
    };

    assert_eq!("ataxx", startup_settings.game);

    let load_network = |path: String, device: Device| -> AtaxxCNNNetwork {
        let graph = startup_settings.tower_shape.to_graph(startup_settings.batch_size as i32);
        AtaxxCNNNetwork::load(path, &graph, startup_settings.batch_size, device)
    };

    let output = |path: &str| AtaxxBinaryOutput::new(path);

    selfplay_server_main(&startup_settings, AtaxxBoard::new_without_gaps, output, load_network);
}

fn print_example_commands() {
    let settings = Settings {
        max_game_length: 400,
        exploration_weight: 2.0,
        random_symmetries: true,
        keep_tree: false,
        temperature: 1.0,
        zero_temp_move_count: 20,
        dirichlet_alpha: 0.25,
        dirichlet_eps: 0.2,
        full_search_prob: 1.0,
        full_iterations: 500,
        part_iterations: 500,
        cache_size: 0,
    };

    println!("{}", serde_json::to_string(&Command::Stop).unwrap());
    println!("{}", serde_json::to_string(&Command::NewSettings(settings)).unwrap());
    println!("{}", serde_json::to_string(&Command::NewNetwork("C:/Documents/Programming/STTT/AlphaZero/data/derp/basic_res_model/params.npz".to_owned())).unwrap());
}

fn _parse_startup() -> StartupSettings {
    let args = std::env::args().collect_vec();
    assert_eq!(2, args.len(), "expected one (json) argument");

    let startup_settings: StartupSettings = serde_json::from_str(&args[1])
        .expect("Failed to parse startup json");
    println!("Startup settings: {:#?}", startup_settings);

    startup_settings
}
