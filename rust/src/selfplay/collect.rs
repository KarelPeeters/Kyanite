use std::io::Write;
use std::time::Instant;

use crossbeam::channel::Receiver;
use sttt::board::{Coord, Player};
use ta::indicators::ExponentialMovingAverage;
use ta::Next;

use crate::selfplay::{Message, Position, Simulation};

pub(super) fn collect(writer: &mut impl Write, game_count: u64, receiver: &Receiver<Message>) {
    //performance metrics
    let mut total_eval_count = 0;
    let mut total_move_count = 0;
    let mut total_game_count = 0;
    let mut total_pos_count = 0;
    let mut eval_throughput = ExponentialMovingAverage::new(5).unwrap();
    let mut move_throughput = ExponentialMovingAverage::new(5).unwrap();
    let mut game_throughput = ExponentialMovingAverage::new(5).unwrap();
    let mut pos_throughput = ExponentialMovingAverage::new(5).unwrap();
    let mut eval_throughput_cached = f64::NAN;
    let mut move_throughput_cached = f64::NAN;

    let start = Instant::now();
    let mut last_print = start;

    for message in receiver {
        let now = Instant::now;
        let elapsed = (now() - start).as_secs_f64();

        //print metrics every second
        if (now() - last_print).as_secs_f64() > 1.0 {
            last_print = now();
            println!("Evals:     {:.2} evals/s => {}", eval_throughput_cached, total_eval_count);
            println!("Moves:     {:.2} moves/s => {}", move_throughput_cached, total_move_count);
            println!("Games:     {:.2} games/s => {} / {}", game_throughput.next((total_game_count as f64) / elapsed), total_game_count, game_count);
            println!("Positions: {:.2} pos/s => {}", pos_throughput.next((total_pos_count as f64) / elapsed), total_pos_count);
            println!();
        }

        //handle incoming message
        match message {
            Message::Simulation(simulation) => {
                total_pos_count += simulation.positions.len() as u64;
                total_game_count += 1;

                append_simulation_to_file(writer, simulation)
                    .expect("Failed to write to output file");
            }
            Message::Counter { moves, evals } => {
                total_eval_count += evals;
                total_move_count += moves;
                eval_throughput_cached = eval_throughput.next((total_eval_count as f64) / elapsed);
                move_throughput_cached = move_throughput.next((total_move_count as f64) / elapsed);
            }
        }

        //we have enough positions, stop
        if total_game_count >= game_count {
            break;
        }
    }
}

const OUTPUT_FORMAT_SIZE: usize = 3 + 3 + 81 + (3 * 81 + 2 * 9);

fn append_simulation_to_file(writer: &mut impl Write, simulation: Simulation) -> std::io::Result<()> {
    let won_by = simulation.won_by;

    let mut full_policy = vec![0.0; 81];
    let mut data = Vec::with_capacity(OUTPUT_FORMAT_SIZE);

    for position in simulation.positions {
        let Position { board, should_store, eval, policy } = position;

        assert_eq!(policy.len(), board.available_moves().count());
        full_policy.fill(0.0);
        for (i, coord) in board.available_moves().enumerate() {
            full_policy[coord.o() as usize] = policy[i];
        }

        data.clear();

        // wdl_final
        data.push((won_by == board.next_player) as u8 as f32);
        data.push((won_by == Player::Neutral) as u8 as f32);
        data.push((won_by == board.next_player.other()) as u8 as f32);

        // wdl_pred
        data.push(eval.win);
        data.push(eval.draw);
        data.push(eval.loss);

        // policy
        data.extend_from_slice(&full_policy);

        // board state
        data.extend(Coord::all().map(|c| board.is_available_move(c) as u8 as f32));
        data.extend(Coord::all().map(|c| (board.tile(c) == board.next_player) as u8 as f32));
        data.extend(Coord::all().map(|c| (board.tile(c) == board.next_player.other()) as u8 as f32));
        data.extend((0..9).map(|om| (board.macr(om) == board.next_player) as u8 as f32));
        data.extend((0..9).map(|om| (board.macr(om) == board.next_player.other()) as u8 as f32));

        assert_eq!(OUTPUT_FORMAT_SIZE, data.len());

        if !should_store {
            write!(writer, "#")?;
        }

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
