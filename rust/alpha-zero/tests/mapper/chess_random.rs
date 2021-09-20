use board_game::board::Board;
use board_game::games::chess::ChessBoard;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use alpha_zero::mapping::chess::ChessStdMapper;

use crate::mapper::test_valid_mapping;

#[test]
#[ignore]
fn main() {
    let mapper = ChessStdMapper;
    let mut rng = SmallRng::seed_from_u64(0);

    let game_count = 10_000;
    for i in 0..game_count {
        println!("Starting game {}/{}", i, game_count);
        let mut board = ChessBoard::default();

        while !board.is_done() {
            test_valid_mapping(mapper, &board);
            board.play(board.random_available_move(&mut rng));
        }
    }
}

