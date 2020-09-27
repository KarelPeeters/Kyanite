use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

use sttt::board::{Board, board_from_compact_string, board_to_compact_string, Coord};
use sttt::bot_game;
use sttt::bot_game::Bot;
use sttt::bots::RandomBot;
use sttt::mcts::heuristic::MacroHeuristic;
use sttt::mcts::MCTSBot;
use sttt::minimax::MiniMaxBot;

fn main() {
    _heuristic_bot_game()
}

fn _time_mcts() {
    let mut board = Board::new();
    board.play(Coord::from_oo(4, 4));
    board.play(Coord::from_oo(4, 0));

    time(|| {
        MCTSBot::new(1_000_000, SmallRng::from_entropy()).play(&board);
    })
}

fn _test_compact_string() {
    let seed: [u8; 16] = Rng::gen(&mut SmallRng::from_entropy());
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

fn _heuristic_bot_game() {
    let res = bot_game::run(
        || MCTSBot::new(50_000, SmallRng::from_entropy()),
        || MCTSBot::new_with_heuristic(50_000, SmallRng::from_entropy(), MacroHeuristic { weight: 1.0 }),
        50,
        true,
    );

    println!("{:?}", res);
}

fn _bot_game() {
    let res = bot_game::run(
        || RandomBot,
        || MCTSBot::new(1000, SmallRng::from_entropy()),
        100,
        true,
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