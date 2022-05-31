use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::create_dir_all;
use std::io::{BufWriter, Write as _};
use std::time::Instant;

use board_game::board::Board;
use flume::Receiver;

use kz_core::mapping::BoardMapper;

use crate::binary_output::BinaryOutput;
use crate::server::protocol::{Evals, GeneratorUpdate, ServerUpdate};

pub fn collector_main<B: Board>(
    game: &str,
    mut writer: BufWriter<impl std::io::Write>,
    muzero: bool,
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
            GeneratorUpdate::RootEvals(evals) => {
                counter.root_evals += evals;
            }
            GeneratorUpdate::ExpandEvals(evals) => {
                counter.expand_evals += evals;
            }
        }

        // periodically print stats
        let now = Instant::now();
        let delta = (now - last_print_time).as_secs_f32();
        if delta >= 1.0 {
            total_games += counter.games;
            total_moves += counter.moves;

            let info = counter
                .to_string(delta, total_moves, total_games, &curr_game_lengths, muzero)
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

    root_evals: Evals,
    expand_evals: Evals,
}

impl Counter {
    fn to_string(
        &self,
        delta: f32,
        total_moves: u64,
        total_games: u64,
        game_lengths: &HashMap<usize, usize>,
        muzero: bool,
    ) -> Result<String, std::fmt::Error> {
        let move_throughput = self.moves as f32 / delta;
        let game_throughput = self.games as f32 / delta;

        let min_game_length = game_lengths.values().copied().min().unwrap_or(0);
        let max_game_length = game_lengths.values().copied().max().unwrap_or(0);
        let mean_game_length = game_lengths.values().copied().sum::<usize>() as f32 / game_lengths.len() as f32;

        let mut result = String::new();
        let f = &mut result;

        writeln!(f, "Selfplay info:")?;
        if muzero {
            write_evals(f, "expand evals", self.expand_evals, delta);
            write_evals(f, "root   evals", self.root_evals, delta);
        } else {
            assert_eq!(self.root_evals, Evals::default());
            write_evals(f, "evals", self.expand_evals, delta);
        }
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

fn write_evals(f: &mut String, name: &str, evals: Evals, delta: f32) {
    let Evals {
        real,
        potential,
        cached,
    } = evals;

    let real_tp = real as f32 / delta;
    let cached_tp = cached as f32 / delta;
    let potential_tp = potential as f32 / delta;

    let hit_rate = (cached as f32) / (real + cached) as f32;
    let fill_rate = (real as f32) / (potential as f32);

    writeln!(
        f,
        "  {}/s: real: {:.1}, cached: {:.1}, potential: {:.1} (hit: {:.2}, fill: {:.2})",
        name, real_tp, cached_tp, potential_tp, hit_rate, fill_rate
    )
    .unwrap();
}
