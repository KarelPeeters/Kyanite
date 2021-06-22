use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use itertools::{Itertools, zip_eq};
use rand::{Rng, SeedableRng, thread_rng};
use rand::rngs::SmallRng;
use rayon::iter::{ParallelBridge, ParallelIterator};

use sttt::board::{Board, board_from_compact_string, board_to_compact_string, Coord, Player};
use sttt::bot_game;
use sttt::bot_game::Bot;
use sttt::bots::RandomBot;
use sttt::mcts::{mcts_build_tree, MCTSBot};
use sttt::minimax::MiniMaxBot;
use sttt::util::lower_process_priority;

fn main() {
    //TODO don't do this when benchmarking!
    lower_process_priority();

    // time(|| mcts_build_tree(&Board::new(), 10_000_000, 2.0, &mut thread_rng()));

    _test_mcts_tree()
}

fn _test_mcts_tree() {
    let tree = mcts_build_tree(&Board::new(), 10_000_000, 2.0, &mut thread_rng());
    tree.print(5);
}

fn _basic_self_play() {
    let iterations = 100_000;

    let mut board = Board::new();
    let mut rng = thread_rng();

    while !board.is_done() {
        println!("{}", board);

        let tree = mcts_build_tree(&board, iterations, 2.0, &mut rng);
        let value = tree.eval().value();

        let x_value = if board.next_player == Player::O { -value } else { value };

        println!("{}", x_value);

        board.play(tree.best_move());
    }

    println!("{}", board);
    println!("{:?}", board.won_by);
}

fn _plot_mismatch_evaluations() {
    let mut board = Board::new();

    let iterations = [100_000, 1_000_000, 10_000_000/*, 100_000_000*/];

    let mut values = vec![vec![]; iterations.len()];

    let mut sign = 1.0;

    while !board.is_done() {
        println!("{}", board);

        let evals = iterations.iter()
            .map(|&iter| {
                let tree = mcts_build_tree(&board, iter, 2.0, &mut thread_rng());
                (tree.best_move(), tree.eval().value())
            })
            .collect_vec();

        for (values, (_, value)) in zip_eq(&mut values, &evals) {
            values.push(sign * value);
        }

        board.play(evals[0].0);
        sign *= -1.0;
    }

    println!("iterations = {:?}", iterations);
    println!("values = {:?}", values);
}

fn _plot_evaluations() {
    let iterations = 100_000;

    let counter = AtomicUsize::new(0);

    let all_values: Vec<Vec<f32>> = (0..100).par_bridge().map(|_| {
        let mut values = vec![];

        let mut board = Board::new();
        while !board.is_done() {
            let mut rng = thread_rng();
            let tree = mcts_build_tree(&board, iterations, 2.0, &mut rng);
            let value = tree.eval().value();

            let x_value = if board.next_player == Player::X { value } else { -value };

            values.push(x_value);
            board.play(tree.best_move());
        }

        let i = counter.fetch_add(1, Ordering::SeqCst);
        println!("{}", i);

        values
    }).collect();

    fn average(values: impl IntoIterator<Item=f32>) -> f32 {
        let mut sum = 0.0;
        let mut count = 0;

        for value in values {
            sum += value;
            count += 1;
        }

        sum / (count as f32)
    }

    println!("{:?}", all_values);

    println!("avg start: {}", average(all_values.iter().map(|v| v.first().unwrap()).copied()));
    println!("avg end: {}", average(all_values.iter().map(|v| v.last().unwrap()).copied()));
}

fn _test_first_move_advantage() {
    let res = bot_game::run(
        || MCTSBot::new(100_000, 2.0, SmallRng::from_entropy()),
        || MCTSBot::new(100_000, 2.0, SmallRng::from_entropy()),
        100, false, Some(10),
    );

    println!("{:?}", res);
}

fn _test_rng() {
    let res = bot_game::run(
        || MCTSBot::new(100_000, 2.0, SmallRng::from_entropy()),
        || MCTSBot::new(100_000, 2.0, thread_rng()),
        100, true, Some(10),
    );

    println!("{:?}", res);
}

fn _time_mcts() {
    let mut board = Board::new();
    board.play(Coord::from_oo(4, 4));
    board.play(Coord::from_oo(4, 0));

    time(|| {
        MCTSBot::new(1_000_000, 2.0, SmallRng::from_entropy()).play(&board);
    })
}

fn _test_compact_string() {
    let seed: [u8; 32] = Rng::gen(&mut SmallRng::from_entropy());
    print!("Seed: {:?}", seed);

    let mut rand = SmallRng::from_seed(seed);

    loop {
        let mut board = Board::new();

        while let Some(mv) = board.random_available_move(&mut rand) {
            board.play(mv);

            let compact_string = board_to_compact_string(&board);
            let rev_board = board_from_compact_string(&compact_string);

            // print!("Board:\n{}\n{:#?}\nRev Board:\n{}\n{:#?}", board, board, rev_board, rev_board);
            assert_eq!(rev_board, board);

            println!("{}", compact_string);
        }
    }
}

fn _test_mm() {
    let board = Board::new();

    let start = Instant::now();
    let mv = MiniMaxBot::new(10).play(&board);
    println!("{:?}", mv);
    println!("{}", start.elapsed().as_millis() as f64 / 1000.0);
}

fn _follow_playout() {
    let moves = [35, 73, 9, 8, 77, 53, 76, 40, 39, 29, 20, 19, 11, 24, 59, 45, 2, 22, 37, 15, 58, 43, 67, 42, 54, 4, 41, 50, 47, 25, 70, 64, 17, 78, 57, 30, 34, 65, 3, 33, 44, 74, 1, 12, 28, 10, 13, 36, 0, 52, 68, 49, 38, 32, 31, ];

    let mut board = Board::new();
    for &mv in moves.iter() {
        board.play(Coord::from_o(mv));
        println!("{}", board);
    }
}

fn _bot_game() {
    let res = bot_game::run(
        || RandomBot::new(SmallRng::from_entropy()),
        || MCTSBot::new(1000, 2.0, SmallRng::from_entropy()),
        100, true, Some(10),
    );

    println!("{:?}", res);
}

#[allow(unused)]
fn time<R, F: FnOnce() -> R>(block: F) -> R {
    let start = Instant::now();
    let result = block();
    println!("Took {:02}s", (Instant::now() - start).as_secs_f32());
    result
}