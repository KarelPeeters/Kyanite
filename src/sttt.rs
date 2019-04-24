#![allow(dead_code)]

use std::collections::HashMap;
use std::time::Instant;

use rand::SeedableRng;

use sttt::{bot_game, hmcts, mcts, minimax};
use sttt::board::{Board, Coord};

fn main() {
    mcts()
}

fn mcts() {
    let mut board = Board::new();
    let mut rng = rand_xorshift::XorShiftRng::from_seed([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

    /*for &mv in &[74, 19, 13, 42, 54, 6, 58, 44, 76, 40, 36, 5, 51, 60, 55, 9, 3, 29, 18, 4, ] {
        board.play(Coord::of_o(mv as u8));
    }*/

    let start = Instant::now();
    let mv = mcts::old_move_mcts(&board, 1_000_000, &mut rng);

    println!("{}", board);
    println!("{:?}", mv);
    println!("took {} ms", start.elapsed().as_millis());
}

//TODO
//replace lastMove by openMask in hash:
//lookup size 81 -> 512
//but bigger chance of catching duplicates!
fn check_hash_collisions() {
    let mut rng = rand::thread_rng();
    let mut map = HashMap::with_capacity(20_000_000);

    let empty = Board::new();
    map.insert(empty.get_hash(), empty);

    for i in 0..100_000 {
        if i % 10_000 == 0 {
            println!("{}:\t{}", i, map.len());
        }

        let mut board = Board::new();

        while let Some(mv) = board.random_available_move(&mut rng) {
            board.play(mv);
            let hash = board.get_hash();

            if let Some(first) = map.get(&hash) {
                let eq = *first == board;
                if !eq {
                    println!("Collision between boards, hash {}", hash);
                    println!("First: \n{}", first);
                    println!("Second: \n{}", board);
                }
            } else {
                map.insert(hash, board.clone());
            }
        }
    }

    println!("Map size: {}", map.len())
}

fn game() {
    let bot_mcts = |board: &Board| {
        mcts::old_move_mcts(board, 1_000_000, &mut rand::thread_rng())
    };
    let bot_mm = |board: &Board| {
        minimax::move_minimax(board, 5)
    };

    let res = bot_game::run(&bot_mcts, &bot_mm, 10, false, &mut rand::thread_rng());

    println!("{:?}", res)
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

