use rand::thread_rng;
use rand_pcg::Pcg32;
use rand_pcg::rand_core::SeedableRng;

use sttt::board::{Board, Coord};
use sttt::mcts::old_move_mcts;
use std::time::Instant;

fn main() {
    _test_mcts()
}

fn _follow_playout() {
    let moves = [35, 73, 9, 8, 77, 53, 76, 40, 39, 29, 20, 19, 11, 24, 59, 45, 2, 22, 37, 15, 58, 43, 67, 42, 54, 4, 41, 50, 47, 25, 70, 64, 17, 78, 57, 30, 34, 65, 3, 33, 44, 74, 1, 12, 28, 10, 13, 36, 0, 52, 68, 49, 38, 32, 31, ];

    let mut board = Board::new();
    for &mv in moves.iter() {
        board.play(Coord::of_o(mv));
        println!("{}", board);
    }
}

fn _test_mcts() {
    let board = Board::new();

    let start = Instant::now();
    println!("{:?}", old_move_mcts(&board, 10 * 1000 * 1000, &mut thread_rng()));
    println!("{}", start.elapsed().as_millis() as f64 / 1000.0);
}