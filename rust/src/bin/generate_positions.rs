use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam::{channel, scope};
use crossbeam::channel::Sender;
use itertools::Itertools;
use rand::{Rng, thread_rng};
use rand::distributions::WeightedIndex;
use sttt::board::{Board, Player, Coord};
use sttt::mcts::heuristic::ZeroHeuristic;
use sttt::mcts::mcts_build_tree;
use std::path::Path;

struct Simulation {
    won_by: Player,
    positions: Vec<Position>,
}

struct Position {
    board: Board,
    child_probabilities: Vec<f32>,
}

fn main() -> std::io::Result<()> {
    sttt::util::lower_process_priority();

    generate_file("../data/new.csv", 200_000, 1_000)?;

    Ok(())
}

fn generate_file(path: &str, min_position_count: usize, mcts_iterations: u64)  -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(&file);

    generate_positions(min_position_count, mcts_iterations, &mut |simulation| {
        append_simulation_to_file(&mut writer, simulation)
    })
}

const OUTPUT_FORMAT_SIZE: usize = 3 + 4 * 81 + 2 * 9;

fn append_simulation_to_file(writer: &mut impl Write, simulation: Simulation) -> std::io::Result<()> {
    let won_by = simulation.won_by;

    let mut full_child_probabilities = vec![0.0; 81];
    let mut data = Vec::new();

    for position in simulation.positions {
        let Position { board, child_probabilities } = position;

        full_child_probabilities.fill(0.0);
        for (i, coord) in board.available_moves().enumerate() {
            full_child_probabilities[coord.om() as usize] = child_probabilities[i];
        }

        data.clear();

        data.push((won_by == board.next_player) as u8 as f32);
        data.push((won_by == Player::Neutral) as u8 as f32);
        data.push((won_by == board.next_player.other()) as u8 as f32);

        data.extend_from_slice(&full_child_probabilities);
        data.extend(Coord::all().map(|c| board.is_available_move(c) as u8 as f32));

        data.extend(Coord::all().map(|c| (board.tile(c) == board.next_player) as u8 as f32));
        data.extend(Coord::all().map(|c| (board.tile(c) == board.next_player.other()) as u8 as f32));

        data.extend((0..9).map(|om| (board.macr(om) == board.next_player) as u8 as f32));
        data.extend((0..9).map(|om| (board.macr(om) == board.next_player.other()) as u8 as f32));

        assert_eq!(OUTPUT_FORMAT_SIZE, data.len());

        for (i, x) in data.iter().enumerate() {
            if i != 0 {
                write!(writer, ",")?;
            }
            write!(writer, "{}", x)?;
        }
        write!(writer, "\n")?;
    }

    write!(writer, "\n")?;

    Ok(())
}

fn generate_positions<F, E>(min_position_count: usize, mcts_iterations: u64, handler: &mut F) -> Result<(), E>
    where F: FnMut(Simulation) -> Result<(), E>
{
    let (sender, receiver) = channel::bounded(1);
    let request_stop = AtomicBool::new(false);

    scope(|s| {
        //spawn a bunch of threads
        for _ in 0..num_cpus::get() {
            let sender = sender.clone();
            s.spawn(|_| generate_positions_thread(sender, &request_stop, mcts_iterations));
        }

        let mut counter = 0;
        for simulation in &receiver {
            counter += simulation.positions.len();

            let progress = counter as f32 / min_position_count as f32;
            println!("Progress: {:.3} = {}/{} positions", progress, counter, min_position_count);

            handler(simulation)?;

            if counter > min_position_count {
                break;
            }
        }

        //tell spawned threads to stop
        request_stop.store(true, Ordering::Relaxed);
        drop(receiver);

        Ok(())

        //scope automatically waits for threads to finish
    }).expect("Threading issue")
}

fn generate_positions_thread(sender: Sender<Simulation>, request_stop: &AtomicBool, mcts_iterations: u64) {
    let mut rng = thread_rng();

    loop {
        let mut positions = Vec::new();
        let mut board = Board::new();

        let final_won_by = loop {
            //early exit
            if request_stop.load(Ordering::Relaxed) {
                return;
            }

            match board.won_by {
                Some(player) => {
                    break player;
                }
                None => {
                    let tree = mcts_build_tree(&board, mcts_iterations, &ZeroHeuristic, &mut rng);

                    let root = &tree[0];
                    let children = root.children().unwrap();

                    let child_probabilities = children.iter().map(|child| {
                        (tree[child].visits as f32) / (root.visits as f32)
                    }).collect_vec();

                    // pick move according to probabilities
                    // TODO temperature?
                    let distr = WeightedIndex::new(&child_probabilities).unwrap();
                    let picked_child = children.get(rng.sample(distr));
                    let picked_move = tree[picked_child].coord;

                    positions.push(Position {
                        board: board.clone(),
                        child_probabilities,
                    });

                    board.play(picked_move);
                }
            }
        };

        if sender.send(Simulation { won_by: final_won_by, positions }).is_err() {
            // receiver disconnected, exit thread
            return;
        }
    }
}
