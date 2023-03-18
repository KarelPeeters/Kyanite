use board_game::board::Board;
use board_game::games::chess::ChessBoard;
use rand::rngs::StdRng;
use rand::SeedableRng;

use kz_core::mapping::chess::{ChessHistoryMapper, ChessLegacyConvPolicyMapper, ChessStdMapper};

use crate::mapper::{test_valid_mapping, test_valid_policy_mapping};

#[test]
#[ignore]
fn history() {
    for length in [0, 1, 2, 8, 16, 32] {
        random_impl(|board| test_valid_mapping(ChessHistoryMapper::new(length), board));
    }
}

#[test]
#[ignore]
fn std() {
    random_impl(|board| test_valid_mapping(ChessStdMapper, board));
}

#[test]
#[ignore]
fn conv() {
    random_impl(|board| test_valid_policy_mapping(ChessLegacyConvPolicyMapper, board));
}

fn random_impl(f: impl Fn(&ChessBoard)) {
    let mut rng = StdRng::seed_from_u64(0);

    let game_count = 10_000;
    for i in 0..game_count {
        println!("Starting game {}/{}", i, game_count);
        let mut board = ChessBoard::default();

        while !board.is_done() {
            f(&board);
            board.play(board.random_available_move(&mut rng).unwrap()).unwrap();
        }
    }
}
