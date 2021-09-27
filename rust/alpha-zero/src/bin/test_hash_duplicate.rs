use board_game::games::chess::ChessBoard;
use board_game::board::Board;
use std::collections::HashMap;
use rand::thread_rng;
use std::collections::hash_map::{DefaultHasher, Entry};
use std::hash::Hasher;

fn main() {
    test_hash_collision(&ChessBoard::default())
}

fn test_hash_collision<B: Board>(start: &B) {
    let mut map = HashMap::<u64, B>::default();
    let mut rng = thread_rng();

    loop {
        let mut board = start.clone();

        while !board.is_done() {
            if map.len() % 1_000_000 == 0 {
                println!("Visited {} unique boards without collision", map.len());
            }

            board.play(board.random_available_move(&mut rng));

            let mut hasher = DefaultHasher::new();
            board.hash(&mut hasher);
            let hash = hasher.finish();

            match map.entry(hash) {
                Entry::Occupied(entry) => {
                    if entry.get() != &board {
                        println!("Found two boards matching hash {}", hash);
                        println!("  '{}'", board);
                        println!("  '{}'", entry.get());
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert(board.clone());
                }
            }
        }
    }
}