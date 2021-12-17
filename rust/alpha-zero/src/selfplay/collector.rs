use std::cmp::{max, Ordering};
use std::collections::BinaryHeap;
use std::fs::create_dir_all;
use std::io::{BufWriter, Write};
use std::time::Instant;

use board_game::board::Board;
use crossbeam::channel::Receiver;
use itertools::Itertools;
use itertools::zip;

use crate::mapping::binary_output::BinaryOutput;
use crate::mapping::BoardMapper;
use crate::selfplay::protocol::{GeneratorUpdate, ServerUpdate};
use crate::selfplay::simulation::Simulation;

pub fn collector_main<B: Board>(
    game: &str,
    mut writer: BufWriter<impl Write>,
    games_per_file: usize,
    first_gen: u32,
    output_folder: &str,
    mapper: impl BoardMapper<B>,
    update_receiver: Receiver<GeneratorUpdate<B>>,
    thread_count: usize,
    reorder_games: bool,
) {
    let new_output = |gen: u32| {
        let path = format!("{}/games_{}", output_folder, gen);
        println!("Collector: start writing to {}", path);
        BinaryOutput::new(path, game, mapper)
            .expect("Error while creating output files")
    };

    create_dir_all(&output_folder)
        .expect("Failed to create output folder");

    let mut curr_gen = first_gen;
    let mut curr_output = new_output(curr_gen);

    let mut curr_game_count = 0;
    let mut estimator = ThroughputEstimator::new();

    // state used to re-order positions per thread
    let mut heaps = (0..thread_count).map(|_| BinaryHeap::new()).collect_vec();
    let mut indices_next_max = vec![(0, 0); thread_count];

    for update in update_receiver {
        match update {
            GeneratorUpdate::Stop => break,
            GeneratorUpdate::FinishedSimulation { thread_id, index: real_index, simulation } => {
                let (next_index, max_index) = &mut indices_next_max[thread_id];
                let index = if reorder_games { real_index } else { *next_index };

                estimator.add_game();
                heaps[thread_id].push(HeapItem { index, simulation });
                *max_index = max(*max_index, index);

                while let Some(item) = heaps[thread_id].peek() {
                    if item.index != *next_index { break; }
                    let item = heaps[thread_id].pop().unwrap();
                    *next_index += 1;

                    curr_output.append(item.simulation)
                        .expect("Error during simulation appending");
                    curr_game_count += 1;

                    if curr_game_count >= games_per_file {
                        curr_output.finish()
                            .expect("Error while finishing output file");

                        let prev_i = curr_gen;
                        curr_gen += 1;
                        curr_game_count = 0;
                        curr_output = new_output(curr_gen);

                        let message = ServerUpdate::FinishedFile { index: prev_i };
                        writer.write_all(serde_json::to_string(&message).unwrap().as_bytes()).unwrap();
                        writer.write(&[b'\n']).unwrap();
                        writer.flush().unwrap();
                    }
                }
            }
            GeneratorUpdate::Progress { cached_evals, real_evals, moves } => {
                estimator.update(real_evals, cached_evals, moves, &heaps, &indices_next_max);
            }
        }
    }

    writer.write_all(serde_json::to_string(&ServerUpdate::Stopped).unwrap().as_bytes()).unwrap();
    writer.write(&[b'\n']).unwrap();
    writer.flush().unwrap()
}

#[derive(Debug)]
struct HeapItem<B> {
    index: u64,
    simulation: Simulation<B>,
}

#[derive(Debug)]
struct ThroughputEstimator {
    last_print_time: Instant,
    real_evals: u64,
    cached_evals: u64,
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

    fn update<B>(
        &mut self, real_evals: u64, cached_evals: u64, moves: u64,
        heaps: &[BinaryHeap<HeapItem<B>>], indices_next_max: &[(u64, u64)],
    ) {
        self.real_evals += real_evals;
        self.cached_evals += cached_evals;
        self.moves += moves;
        self.total_moves += moves;

        let now = Instant::now();
        let delta = (now - self.last_print_time).as_secs_f32();

        if delta >= 1.0 {
            self.last_print_time = now;
            let real_eval_throughput = self.real_evals as f32 / delta;
            let cached_eval_throughput = self.cached_evals as f32 / delta;
            let move_throughput = self.moves as f32 / delta;
            let game_throughput = self.games as f32 / delta;

            let heap_info = zip(heaps, indices_next_max).map(|(h, &(next, max))| {
                let min = h.peek().map_or(0, |item| item.index);
                (h.len(), next, min, max)
            }).collect_vec();

            println!(
                "Thoughput: {:.2} evals/s, {:.2} cached evals/s, {:.2} moves/s => {} moves {:.2} games/s => {} games",
                real_eval_throughput, cached_eval_throughput, move_throughput, self.total_moves, game_throughput, self.total_games
            );
            println!("   cache hit rate: {}", cached_eval_throughput / (cached_eval_throughput + real_eval_throughput));
            println!("   reorder heaps (size, i_next, i_min, i_max): {:?}", heap_info);

            self.real_evals = 0;
            self.cached_evals = 0;
            self.moves = 0;
            self.games = 0;
        }
    }
}

impl<B> Eq for HeapItem<B> {}

impl<B> PartialEq<Self> for HeapItem<B> {
    fn eq(&self, other: &Self) -> bool {
        self.index.eq(&other.index)
    }
}

impl<B> Ord for HeapItem<B> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.index.cmp(&other.index).reverse()
    }
}

impl<B> PartialOrd for HeapItem<B> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}