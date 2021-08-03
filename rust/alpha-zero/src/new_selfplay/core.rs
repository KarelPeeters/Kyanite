use serde::{Deserialize, Serialize};

use crate::selfplay::Simulation;
use crate::network::tower_shape::TowerShape;

#[derive(Debug, Serialize, Deserialize)]
pub struct StartupSettings {
    pub game: String,
    pub output_folder: String,
    pub threads_per_device: usize,
    pub batch_size: usize,
    pub games_per_file: usize,
    pub tower_shape: TowerShape,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Command {
    Stop,
    NewSettings(Settings),
    NewNetwork(String),
}

#[derive(Debug)]
pub enum GeneratorUpdate<B> {
    Stop,

    FinishedSimulation(Simulation<B>),

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
pub enum ServerUpdate {
    Stopped,
    FinishedFile,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Settings {
    // self-play game affecting settings
    pub max_game_length: u64,
    pub exploration_weight: f32,

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