use std::time::Instant;

use rand::SeedableRng;

use sttt::{bot_game, mcts, minimax};
use sttt::board::{Board, Coord};

fn main() {
    _mcts();
}

fn game() {
    let bot_mcts = |board: &Board| {
        mcts::move_mcts(board, 1_000_000, &mut rand::thread_rng())
    };
    let bot_mm = |board: &Board| {
        minimax::move_minimax(board, 5)
    };

    let res = bot_game::run(&bot_mcts, &bot_mm, 10, false, &mut rand::thread_rng());

    println!("{:?}", res)
}

fn _mcts() {
    let mut board = Board::new();
    let mut rng = rand_xorshift::XorShiftRng::from_seed([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

    for &mv in &[74, 19, 13, 42, 54, 6, 58, 44, 76, 40, 36, 5, 51, 60, 55, 9, 3, 29, 18, 4, ] {
        board.play(Coord::of_o(mv as u8));
    }

    let start = Instant::now();
    let mv = mcts::move_mcts(&board, 1_000_000, &mut rng);

    println!("{}", board);
    println!("{:?}", mv);
    println!("took {} ms", start.elapsed().as_millis());
}

fn _bench() {
    let mut board = Board::new();
    let mut _rng = rand::thread_rng();

    let start = Instant::now();
    let mut i = 0;

    while !board.is_done() {
        println!("{}", i);
        i += 1;

        let mov = minimax::move_minimax(&board, 11).expect("Board isn't done, there should be a move");
        board.play(mov);
    }

    println!("{:?}", board.won_by);

    println!("took {} ms", start.elapsed().as_millis());
}

