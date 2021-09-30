use alpha_zero::mapping::chess::ChessStdMapper;
use alpha_zero::mapping::PolicyMapper;
use board_game::games::chess::ChessBoard;
use board_game::board::{BoardAvailableMoves, Board};
use internal_iterator::InternalIterator;
use itertools::Itertools;
use rand::thread_rng;

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

    let mut visited = vec![(0, None); policy_size];
    for i in 0..100_000 {
        if i % 1000 == 0 {
            println!("{}", i);
        }
        let mut board = ChessBoard::default();

        while !board.is_done() {
            let mut count = 0;

            board.available_moves().for_each(|mv| {
                if let Some(index) = mapper.move_to_index(&board, mv) {
                    let prev = &visited[index];
                    visited[index] = (prev.0 + 1, Some(mv));
                }

                count += 1;
            });

            println!("Available moves: {}", count);

            board.play(board.random_available_move(&mut rng));
        }
    }

    println!("visited min/max: {:?}", visited.iter().filter(|&&n| n.0 != 0).minmax());
    println!("visited count {}", visited.iter().filter(|&&n| n.0 != 0).count());

    for (i, &(count, mv)) in visited.iter().enumerate() {
        if let Some(mv) = mv {
            println!("{}: {:?}, {}", i,  mv.to_string(), count);
        }
    }
}