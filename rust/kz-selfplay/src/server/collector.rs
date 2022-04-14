use std::cmp::max;
use std::fs::create_dir_all;
use std::io::{BufWriter, Write};
use std::time::Instant;

use board_game::board::Board;
use crossbeam::channel::Receiver;
use std::fmt::Write as fmtWrite;

use kz_core::mapping::BoardMapper;

use crate::binary_output::BinaryOutput;
use crate::server::protocol::{GeneratorUpdate, ServerUpdate};

pub fn collector_main<B: Board>(
    game: &str,
    mut writer: BufWriter<impl Write>,
    games_per_file: usize,
    first_gen: u32,
    output_folder: &str,
    mapper: impl BoardMapper<B>,
    update_receiver: Receiver<GeneratorUpdate<B>>,
    thread_count: usize,
) {
    let new_output = |gen: u32| {
        let path = format!("{}/games_{}", output_folder, gen);
        println!("Collector: start writing to {}", path);
        BinaryOutput::new(path, game, mapper).expect("Error while creating output files")
    };

    create_dir_all(&output_folder).expect("Failed to create output folder");

    let mut curr_gen = first_gen;
    let mut curr_output = new_output(curr_gen);

    let mut curr_game_count = 0;
    let mut estimator = ThroughputEstimator::new();

    let mut indices_min_max = vec![(0, 0); thread_count];

    for update in update_receiver {
        match update {
            GeneratorUpdate::Stop => break,
            GeneratorUpdate::StartedSimulations {
                thread_id,
                next_index: last_index,
            } => {
                let next_index = &mut indices_min_max[thread_id].0;
                assert!(last_index >= *next_index);
                *next_index = last_index + 1;
            }
            GeneratorUpdate::FinishedSimulation {
                thread_id,
                index,
                simulation,
            } => {
                let max_index = &mut indices_min_max[thread_id].1;
                *max_index = max(*max_index, index);

                estimator.add_game();

                curr_output
                    .append(&simulation)
                    .expect("Error during simulation appending");
                curr_game_count += 1;

                if curr_game_count >= games_per_file {
                    curr_output.finish().expect("Error while finishing output file");

                    let prev_i = curr_gen;
                    curr_gen += 1;
                    curr_game_count = 0;
                    curr_output = new_output(curr_gen);

                    let message = ServerUpdate::FinishedFile { index: prev_i };
                    writer
                        .write_all(serde_json::to_string(&message).unwrap().as_bytes())
                        .unwrap();
                    writer.write_all(&[b'\n']).unwrap();
                    writer.flush().unwrap();
                }
            }
            GeneratorUpdate::Progress {
                cached_evals,
                real_evals,
                root_evals,
                moves,
            } => {
                estimator.update(real_evals, cached_evals, root_evals, moves, &indices_min_max);
            }
        }
    }

    writer
        .write_all(serde_json::to_string(&ServerUpdate::Stopped).unwrap().as_bytes())
        .unwrap();
    writer.write_all(&[b'\n']).unwrap();
    writer.flush().unwrap()
}

#[derive(Debug)]
struct ThroughputEstimator {
    last_print_time: Instant,
    real_evals: u64,
    cached_evals: u64,
    root_evals: u64,
    moves: u64,
    games: u64,
    total_moves: u64,
    total_games: u64,
}

impl ThroughputEstimator {
    fn new() -> Self {
        ThroughputEstimator {
            last_print_time: Instant::now(),
            real_evals: 0,
            cached_evals: 0,
            root_evals: 0,
            moves: 0,
            games: 0,
            total_moves: 0,
            total_games: 0,
        }
    }

    fn add_game(&mut self) {
        self.games += 1;
        self.total_games += 1;
    }

    fn update(
        &mut self,
        real_evals: u64,
        cached_evals: u64,
        root_evals: u64,
        moves: u64,
        indices_next_max: &[(u64, u64)],
    ) {
        self.real_evals += real_evals;
        self.cached_evals += cached_evals;
        self.root_evals += root_evals;
        self.moves += moves;
        self.total_moves += moves;

        let now = Instant::now();
        let delta = (now - self.last_print_time).as_secs_f32();

        if delta >= 1.0 {
            self.last_print_time = now;
            let real_eval_throughput = self.real_evals as f32 / delta;
            let cached_eval_throughput = self.cached_evals as f32 / delta;
            let root_eval_throughput = self.root_evals as f32 / delta;
            let move_throughput = self.moves as f32 / delta;
            let game_throughput = self.games as f32 / delta;

            let cache_hit_rate = cached_eval_throughput / (cached_eval_throughput + real_eval_throughput);

            let mut info = String::new();

            writeln!(&mut info, "Selfplay info:").unwrap();
            writeln!(
                &mut info,
                "  {:.2} gpu evals/s, {:.2} cached evals/s, (hit rate {:.2})",
                real_eval_throughput, cached_eval_throughput, cache_hit_rate
            )
            .unwrap();
            writeln!(&mut info, "  {:.2} root evals/s", root_eval_throughput).unwrap();
            writeln!(
                &mut info,
                "  {:.2} moves/s => {} moves {:.2} games/s => {} games",
                move_throughput, self.total_moves, game_throughput, self.total_games
            )
            .unwrap();
            writeln!(&mut info, "  indices: {:?}", indices_next_max).unwrap();

            print!("{}", info);

            self.real_evals = 0;
            self.cached_evals = 0;
            self.root_evals = 0;
            self.moves = 0;
            self.games = 0;
        }
    }
}
