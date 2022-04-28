use std::collections::HashMap;
use std::fs::create_dir_all;
use std::io::{BufWriter, Write};
use std::time::Instant;

use board_game::board::Board;
use flume::Receiver;
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
) {
    let new_output = |gen: u32| {
        let path = format!("{}/games_{}", output_folder, gen);
        println!("Collector: start writing to {}", path);
        BinaryOutput::new(path, game, mapper).expect("Error while creating output files")
    };

    create_dir_all(&output_folder).expect("Failed to create output folder");

    let mut curr_gen = first_gen;
    let mut curr_output = new_output(curr_gen);

    let mut total_games = 0;
    let mut total_moves = 0;
    let mut counter = Counter::default();

    let mut last_print_time = Instant::now();
    let mut curr_game_lengths = HashMap::new();

    for update in update_receiver {
        match update {
            GeneratorUpdate::Stop => break,

            GeneratorUpdate::StartedSimulation { generator_id } => {
                curr_game_lengths.insert(generator_id, 0);
            }

            GeneratorUpdate::FinishedMove {
                generator_id,
                curr_game_length,
            } => {
                curr_game_lengths.insert(generator_id, curr_game_length);
                counter.moves += 1;
            }

            GeneratorUpdate::FinishedSimulation {
                generator_id: _,
                simulation,
            } => {
                counter.games += 1;

                // write file to disk, possibly starting a new generation
                curr_output
                    .append(&simulation)
                    .expect("Error during simulation appending");

                if curr_output.game_count() >= games_per_file {
                    curr_output.finish().expect("Error while finishing output file");

                    let prev_i = curr_gen;
                    curr_gen += 1;
                    curr_output = new_output(curr_gen);

                    let message = ServerUpdate::FinishedFile { index: prev_i };
                    writer
                        .write_all(serde_json::to_string(&message).unwrap().as_bytes())
                        .unwrap();
                    writer.write_all(&[b'\n']).unwrap();
                    writer.flush().unwrap();
                }
            }
            GeneratorUpdate::Evals {
                cached_evals,
                real_evals,
                root_evals,
            } => {
                counter.cached_evals += cached_evals;
                counter.real_evals += real_evals;
                counter.root_evals += root_evals;
            }
        }

        // periodically print stats
        let now = Instant::now();
        let delta = (now - last_print_time).as_secs_f32();
        if delta >= 1.0 {
            total_games += counter.games;
            total_moves += counter.moves;

            let info = counter
                .to_string(delta, total_moves, total_games, &curr_game_lengths)
                .unwrap();
            print!("{}", info);

            counter = Counter::default();
            last_print_time = now;
        }
    }

    writer
        .write_all(serde_json::to_string(&ServerUpdate::Stopped).unwrap().as_bytes())
        .unwrap();
    writer.write_all(&[b'\n']).unwrap();
    writer.flush().unwrap()
}

#[derive(Default, Debug)]
struct Counter {
    moves: u64,
    games: u64,

    cached_evals: u64,
    real_evals: u64,
    root_evals: u64,
}

impl Counter {
    fn to_string(
        &self,
        delta: f32,
        total_moves: u64,
        total_games: u64,
        game_lengths: &HashMap<usize, usize>,
    ) -> Result<String, std::fmt::Error> {
        let real_eval_throughput = self.real_evals as f32 / delta;
        let cached_eval_throughput = self.cached_evals as f32 / delta;
        let root_eval_throughput = self.root_evals as f32 / delta;
        let move_throughput = self.moves as f32 / delta;
        let game_throughput = self.games as f32 / delta;

        let cache_hit_rate = cached_eval_throughput / (cached_eval_throughput + real_eval_throughput);

        let min_game_length = game_lengths.values().copied().min().unwrap_or(0);
        let max_game_length = game_lengths.values().copied().max().unwrap_or(0);
        let mean_game_length = game_lengths.values().copied().sum::<usize>() as f32 / game_lengths.len() as f32;

        let mut result = String::new();
        let f = &mut result;

        writeln!(f, "Selfplay info:")?;
        writeln!(
            f,
            "  {:.2} gpu evals/s, {:.2} cached evals/s, (hit rate {:.2})",
            real_eval_throughput, cached_eval_throughput, cache_hit_rate
        )?;
        writeln!(f, "  {:.2} root evals/s", root_eval_throughput)?;
        writeln!(
            f,
            "  {:.2} moves/s => {} moves {:.2} games/s => {} games",
            move_throughput, total_moves, game_throughput, total_games
        )?;
        writeln!(
            f,
            "  game lengths: min {} max {} mean {:.2}",
            min_game_length, max_game_length, mean_game_length
        )?;

        Ok(result)
    }
}
