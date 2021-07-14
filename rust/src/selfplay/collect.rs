use std::time::Instant;

use crossbeam::channel::Receiver;
use sttt::board::Board;
use ta::indicators::ExponentialMovingAverage;
use ta::Next;

use crate::selfplay::{Message, Simulation};

pub(super) fn collect<B: Board>(
    game_count: u64,
    receiver: &Receiver<Message<B>>,
    mut output: impl FnMut(Simulation<B>) -> (),
) {
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

                output(simulation);
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
