use std::cmp::{max, min};

use board_game::board::{Board, BoardAvailableMoves};
use board_game::games::chess::ChessBoard;
use internal_iterator::InternalIterator;
use itertools::Itertools;
use rand::thread_rng;

use alpha_zero::mapping::chess::ChessStdMapper;
use alpha_zero::mapping::PolicyMapper;

fn main() {
    println!("Hello world");
    let mut rng = thread_rng();

    let mapper = ChessStdMapper;
    let policy_size = ChessStdMapper::POLICY_SIZE;
    let policy_shape = ChessStdMapper::POLICY_SIZE;

    let knight_moves = 16 * 8 + 16 * 6 + 20 * 4 + 8 * 3 + 4 * 2;
    let queen_moves = 21 * (8 * 8 - 7 * 7) + 23 * (7 * 7 - 6 * 6) + 25 * (4 * 4 - 2 * 2) + 27 * 2 * 2;
    let underpromotion_moves = 6 * 3 * 3 + 2 * 2 * 3;
    let distinct_moves = knight_moves + queen_moves + underpromotion_moves;

    println!("possible: {}", distinct_moves);
    println!("total: {}", policy_size);
    println!("total shape: {:?}", policy_shape);

    let mut max_available_moves = 0;
    let mut min_available_moves = usize::MAX;
    let mut total_available_moves = 0;
    let mut position_count = 0;

    let mut visited = vec![(0, None); policy_size];
    let game_count = 100_000;

    for i in 0..game_count {
        if i % 1000 == 0 {
            println!("{}/{}", i, game_count);
        }
        let mut board = ChessBoard::default();

        while !board.is_done() {
            let mut available_moves = 0;
            position_count += 1;

            board.available_moves().for_each(|mv| {
                if let Some(index) = mapper.move_to_index(&board, mv) {
                    let prev = &visited[index];
                    visited[index] = (prev.0 + 1, Some(mv));
                }

                available_moves += 1;
            });

            min_available_moves = min(min_available_moves, available_moves);
            max_available_moves = max(max_available_moves, available_moves);
            total_available_moves += available_moves;

            board.play(board.random_available_move(&mut rng));
        }
    }

    println!("visited min/max: {:?}", visited.iter().filter(|&&n| n.0 != 0).minmax());
    println!("visited count {}", visited.iter().filter(|&&n| n.0 != 0).count());

    println!("available moves:");
    println!("  min: {}", min_available_moves);
    println!("  avg: {:.2}", total_available_moves as f32 / position_count as f32);
    println!("  max: {}", max_available_moves);

    for (i, &(count, mv)) in visited.iter().enumerate() {
        if let Some(mv) = mv {
            println!("{}: {:?}, {}", i, mv.to_string(), count);
        }
    }

    let occupied = visited.iter()
        .enumerate()
        .filter(|(_, (c, _))| *c != 0)
        .map(|(i, _)| i)
        .collect_vec();

    println!("Occupied: {:?}", occupied);
}