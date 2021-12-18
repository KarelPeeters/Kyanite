use board_game::board::Board;
use serde::{Deserialize, Serialize};

use crate::selfplay::simulation::Simulation;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupSettings {
    pub game: String,
    pub output_folder: String,
    pub threads_per_device: usize,
    pub batch_size: usize,
    pub games_per_gen: usize,
    pub first_gen: u32,
    pub reorder_games: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum Command {
    StartupSettings(StartupSettings),
    NewSettings(Settings),
    NewNetwork(String),
    WaitForNewNetwork,
    Stop,
}

#[derive(Debug)]
pub enum GeneratorUpdate<B: Board> {
    Stop,

    FinishedSimulation {
        thread_id: usize,
        index: u64,
        simulation: Simulation<B>,
    },

    // all values since the last progress update
    Progress {
        // the number of evaluations that hit the cache
        cached_evals: u64,
        // the number of evaluations that did not hit the cache
        real_evals: u64,
        // the number of moves played
        moves: u64,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ServerUpdate {
    Stopped,
    FinishedFile { index: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    // self-play game affecting settings
    pub max_game_length: i64,
    pub exploration_weight: f32,
    pub use_value: bool,

    pub random_symmetries: bool,
    pub keep_tree: bool,

    pub temperature: f32,
    pub zero_temp_move_count: u32,

    pub dirichlet_alpha: f32,
    pub dirichlet_eps: f32,

    pub full_search_prob: f64,
    pub full_iterations: u64,
    pub part_iterations: u64,

    // performance
    pub cache_size: usize,
}