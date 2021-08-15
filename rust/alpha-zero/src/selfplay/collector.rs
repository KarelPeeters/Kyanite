use std::io::{BufWriter, Write};
use std::time::Instant;

use crossbeam::channel::Receiver;

use board_game::board::Board;

use crate::selfplay::core::Output;
use crate::selfplay::protocol::{GeneratorUpdate, ServerUpdate};

pub fn collector_main<B: Board, O: Output<B>>(
    mut writer: BufWriter<impl Write>,
    games_per_file: usize,
    first_gen: u32,
    output_folder: &str,
    output: impl Fn(&str) -> O,
    update_receiver: Receiver<GeneratorUpdate<B>>,
) {
    let mut curr_gen = first_gen;
    let curr_path = format!("{}/games_{}.bin", output_folder, curr_gen);
    let mut curr_output = output(&curr_path);
    println!("Collector: start writing to {}", curr_path);

    let mut curr_game_count = 0;
    let mut estimator = ThroughputEstimator::new();

    for update in update_receiver {
        match update {
            GeneratorUpdate::Stop => break,
            GeneratorUpdate::FinishedSimulation(simulation) => {
                estimator.add_game();

                curr_output.append(simulation);
                curr_game_count += 1;

                if curr_game_count >= games_per_file {
                    let prev_i = curr_gen;
                    curr_gen += 1;
                    curr_game_count = 0;

                    let curr_path = format!("{}/games_{}.bin", output_folder, curr_gen);
                    curr_output = output(&curr_path);
                    println!("Collector: start writing to {}", curr_path);

                    let message = ServerUpdate::FinishedFile { index: prev_i };
                    writer.write_all(serde_json::to_string(&message).unwrap().as_bytes()).unwrap();
                    writer.write(&[b'\n']).unwrap();
                    writer.flush().unwrap();
                }
            }
            GeneratorUpdate::Progress { cached_evals, real_evals, moves } => {
                estimator.update(real_evals, cached_evals, moves);
            }
        }
    }

    writer.write_all(serde_json::to_string(&ServerUpdate::Stopped).unwrap().as_bytes()).unwrap();
    writer.write(&[b'\n']).unwrap();
    writer.flush().unwrap()
}

struct ThroughputEstimator {
    last_print_time: Instant,
    real_evals: u64,
    cached_evals: u64,
    moves: u64,
    games: u64,
    total_games: u64,
}

impl ThroughputEstimator {
    fn new() -> Self {
        ThroughputEstimator {
            last_print_time: Instant::now(),
            real_evals: 0,
            cached_evals: 0,
            moves: 0,
            games: 0,
            total_games: 0,
        }
    }

    fn add_game(&mut self) {
        self.games += 1;
        self.total_games += 1;
    }

    fn update(&mut self, real_evals: u64, cached_evals: u64, moves: u64) {
        self.real_evals += real_evals;
        self.cached_evals += cached_evals;
        self.moves += moves;

        let now = Instant::now();
        let delta = (now - self.last_print_time).as_secs_f32();

        if delta >= 1.0 {
            self.last_print_time = now;
            let real_eval_throughput = self.real_evals as f32 / delta;
            let cached_eval_throughput = self.cached_evals as f32 / delta;
            let moves_throughput = self.moves as f32 / delta;
            let game_throughput = self.games as f32 / delta;

            println!(
                "Thoughput: {} evals/s, {} cached evals/s, {} moves/s, {} games/s, {} games",
                real_eval_throughput, cached_eval_throughput, moves_throughput, game_throughput, self.total_games
            );
            println!("   cache hit rate: {}", cached_eval_throughput / (cached_eval_throughput + real_eval_throughput));

            self.real_evals = 0;
            self.cached_evals = 0;
            self.moves = 0;
            self.games = 0;
        }
    }
}
