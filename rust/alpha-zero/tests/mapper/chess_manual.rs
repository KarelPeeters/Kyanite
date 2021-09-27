use std::str::FromStr;

use board_game::board::Board;
use board_game::games::chess::{ChessBoard, Rules};
use chess::{ChessMove, Piece, Square};

use alpha_zero::mapping::chess::{ChessStdMapper, ClassifiedPovMove};
use alpha_zero::mapping::PolicyMapper;
use alpha_zero::util::display_option;

use crate::mapper::test_valid_mapping;

#[test]
fn basic_board_mapping() {
    test_valid_mapping(ChessStdMapper, &ChessBoard::default());
}

#[test]
fn queen_distance_white() {
    // mostly empty with white queen on A1
    let board = board("8/8/8/6k1/8/6K1/8/Q7 w - - 0 1");

    test_policy_pairs(&board, &[
        (Some(0 * 64), Some(ChessMove::new(Square::A1, Square::A2, None))),
        (Some(1 * 64), Some(ChessMove::new(Square::A1, Square::A3, None))),
        (Some(2 * 64), Some(ChessMove::new(Square::A1, Square::A4, None))),
        (Some(3 * 64), Some(ChessMove::new(Square::A1, Square::A5, None))),
        (Some(4 * 64), Some(ChessMove::new(Square::A1, Square::A6, None))),
        (Some(5 * 64), Some(ChessMove::new(Square::A1, Square::A7, None))),
        (Some(6 * 64), Some(ChessMove::new(Square::A1, Square::A8, None))),
    ]);
}

#[test]
fn queen_distance_black() {
    // mostly empty with black queen on A8
    let board = board("q7/8/8/6k1/8/6K1/8/8 b - - 0 1");

    test_policy_pairs(&board, &[
        (Some(0 * 64), Some(ChessMove::new(Square::A8, Square::A7, None))),
        (Some(1 * 64), Some(ChessMove::new(Square::A8, Square::A6, None))),
        (Some(2 * 64), Some(ChessMove::new(Square::A8, Square::A5, None))),
        (Some(3 * 64), Some(ChessMove::new(Square::A8, Square::A4, None))),
        (Some(4 * 64), Some(ChessMove::new(Square::A8, Square::A3, None))),
        (Some(5 * 64), Some(ChessMove::new(Square::A8, Square::A2, None))),
        (Some(6 * 64), Some(ChessMove::new(Square::A8, Square::A1, None))),
    ]);
}

#[test]
fn queen_direction_white() {
    // mostly empty with white queen on D4
    let board = board("8/8/6k1/8/3Q4/6K1/8/8 w - - 0 1");

    let d4 = Square::D4.to_index();
    test_policy_pairs(&board, &[
        (Some(0 * 7 * 64 + d4), Some(ChessMove::new(Square::D4, Square::D5, None))),
        (Some(1 * 7 * 64 + d4), Some(ChessMove::new(Square::D4, Square::E5, None))),
        (Some(2 * 7 * 64 + d4), Some(ChessMove::new(Square::D4, Square::E4, None))),
        (Some(3 * 7 * 64 + d4), Some(ChessMove::new(Square::D4, Square::E3, None))),
        (Some(4 * 7 * 64 + d4), Some(ChessMove::new(Square::D4, Square::D3, None))),
        (Some(5 * 7 * 64 + d4), Some(ChessMove::new(Square::D4, Square::C3, None))),
        (Some(6 * 7 * 64 + d4), Some(ChessMove::new(Square::D4, Square::C4, None))),
        (Some(7 * 7 * 64 + d4), Some(ChessMove::new(Square::D4, Square::C5, None))),
    ])
}

#[test]
fn queen_direction_black() {
    // mostly empty with black queen on D5
    let board = board("8/8/6k1/3q4/8/6K1/8/8 b - - 0 1");

    let d4 = Square::D4.to_index();
    test_policy_pairs(&board, &[
        (Some(0 * 7 * 64 + d4), Some(ChessMove::new(Square::D5, Square::D4, None))),
        (Some(1 * 7 * 64 + d4), Some(ChessMove::new(Square::D5, Square::E4, None))),
        (Some(2 * 7 * 64 + d4), Some(ChessMove::new(Square::D5, Square::E5, None))),
        (Some(3 * 7 * 64 + d4), Some(ChessMove::new(Square::D5, Square::E6, None))),
        (Some(4 * 7 * 64 + d4), Some(ChessMove::new(Square::D5, Square::D6, None))),
        (Some(5 * 7 * 64 + d4), Some(ChessMove::new(Square::D5, Square::C6, None))),
        (Some(6 * 7 * 64 + d4), Some(ChessMove::new(Square::D5, Square::C5, None))),
        (Some(7 * 7 * 64 + d4), Some(ChessMove::new(Square::D5, Square::C4, None))),
    ])
}

#[test]
fn knight_direction_white() {
    // mostly empty with white knight on D4
    let board = board("8/8/6k1/8/3N4/6K1/8/8 w - - 0 1");

    let d4 = Square::D4.to_index();
    test_policy_pairs(&board, &[
        (Some(56 * 64 + d4), Some(ChessMove::new(Square::D4, Square::E6, None))),
        (Some(57 * 64 + d4), Some(ChessMove::new(Square::D4, Square::F5, None))),
        (Some(58 * 64 + d4), Some(ChessMove::new(Square::D4, Square::F3, None))),
        (Some(59 * 64 + d4), Some(ChessMove::new(Square::D4, Square::E2, None))),
        (Some(60 * 64 + d4), Some(ChessMove::new(Square::D4, Square::C2, None))),
        (Some(61 * 64 + d4), Some(ChessMove::new(Square::D4, Square::B3, None))),
        (Some(62 * 64 + d4), Some(ChessMove::new(Square::D4, Square::B5, None))),
        (Some(63 * 64 + d4), Some(ChessMove::new(Square::D4, Square::C6, None))),
    ])
}

#[test]
fn knight_direction_black() {
    // mostly empty with black knight on D5
    let board = board("8/8/6k1/3n4/8/6K1/8/8 b - - 0 1");

    let d4 = Square::D4.to_index();
    test_policy_pairs(&board, &[
        (Some(56 * 64 + d4), Some(ChessMove::new(Square::D5, Square::E3, None))),
        (Some(57 * 64 + d4), Some(ChessMove::new(Square::D5, Square::F4, None))),
        (Some(58 * 64 + d4), Some(ChessMove::new(Square::D5, Square::F6, None))),
        (Some(59 * 64 + d4), Some(ChessMove::new(Square::D5, Square::E7, None))),
        (Some(60 * 64 + d4), Some(ChessMove::new(Square::D5, Square::C7, None))),
        (Some(61 * 64 + d4), Some(ChessMove::new(Square::D5, Square::B6, None))),
        (Some(62 * 64 + d4), Some(ChessMove::new(Square::D5, Square::B4, None))),
        (Some(63 * 64 + d4), Some(ChessMove::new(Square::D5, Square::C3, None))),
    ])
}

#[test]
fn white_potential_promotions() {
    // lots of promotion opportunities for white
    let board = board("r1r5/1P4R1/5RNP/2k5/5K2/pnr5/1r4p1/5R1R w - - 0 1");

    test_policy_pairs(&board, &[
        // rook, no promotion
        (Some((0 * 7 + 1) * 64 + Square::F6.to_index()), Some(ChessMove::new(Square::F6, Square::F8, None))),
        (Some((0 * 7 + 0) * 64 + Square::G7.to_index()), Some(ChessMove::new(Square::G7, Square::G8, None))),
        // knight, no promotion
        (Some(63 * 64 + Square::G6.to_index()), Some(ChessMove::new(Square::G6, Square::F8, None))),
        (Some(56 * 64 + Square::G6.to_index()), Some(ChessMove::new(Square::G6, Square::H8, None))),

        // promotion to queen
        (Some((7 * 7 + 0) * 64 + Square::B7.to_index()), Some(ChessMove::new(Square::B7, Square::A8, Some(Piece::Queen)))),
        (Some((0 * 7 + 0) * 64 + Square::B7.to_index()), Some(ChessMove::new(Square::B7, Square::B8, Some(Piece::Queen)))),
        (Some((1 * 7 + 0) * 64 + Square::B7.to_index()), Some(ChessMove::new(Square::B7, Square::C8, Some(Piece::Queen)))),

        // underpromotion
        (Some(64 * 64 + Square::B7.to_index()), Some(ChessMove::new(Square::B7, Square::A8, Some(Piece::Rook)))),
        (Some(67 * 64 + Square::B7.to_index()), Some(ChessMove::new(Square::B7, Square::B8, Some(Piece::Rook)))),
        (Some(70 * 64 + Square::B7.to_index()), Some(ChessMove::new(Square::B7, Square::C8, Some(Piece::Rook)))),
        (Some(65 * 64 + Square::B7.to_index()), Some(ChessMove::new(Square::B7, Square::A8, Some(Piece::Bishop)))),
        (Some(68 * 64 + Square::B7.to_index()), Some(ChessMove::new(Square::B7, Square::B8, Some(Piece::Bishop)))),
        (Some(71 * 64 + Square::B7.to_index()), Some(ChessMove::new(Square::B7, Square::C8, Some(Piece::Bishop)))),
        (Some(66 * 64 + Square::B7.to_index()), Some(ChessMove::new(Square::B7, Square::A8, Some(Piece::Knight)))),
        (Some(69 * 64 + Square::B7.to_index()), Some(ChessMove::new(Square::B7, Square::B8, Some(Piece::Knight)))),
        (Some(72 * 64 + Square::B7.to_index()), Some(ChessMove::new(Square::B7, Square::C8, Some(Piece::Knight)))),
    ])
}

#[test]
fn black_potential_promotions() {
    // lots of promotion opportunities for black
    let board = board("r1r5/1P4R1/5RNP/2k5/5K2/pnr5/1r4p1/5R1R b - - 0 1");

    // careful, move indices are from the POV of black!
    test_policy_pairs(&board, &[
        // rook, no promotion
        (Some((0 * 7 + 1) * 64 + Square::C6.to_index()), Some(ChessMove::new(Square::C3, Square::C1, None))),
        (Some((0 * 7 + 0) * 64 + Square::B7.to_index()), Some(ChessMove::new(Square::B2, Square::B1, None))),
        // knight, no promotion
        (Some(56 * 64 + Square::B6.to_index()), Some(ChessMove::new(Square::B3, Square::C1, None))),
        (Some(63 * 64 + Square::B6.to_index()), Some(ChessMove::new(Square::B3, Square::A1, None))),

        // promotion to queen
        (Some((7 * 7 + 0) * 64 + Square::G7.to_index()), Some(ChessMove::new(Square::G2, Square::F1, Some(Piece::Queen)))),
        (Some((0 * 7 + 0) * 64 + Square::G7.to_index()), Some(ChessMove::new(Square::G2, Square::G1, Some(Piece::Queen)))),
        (Some((1 * 7 + 0) * 64 + Square::G7.to_index()), Some(ChessMove::new(Square::G2, Square::H1, Some(Piece::Queen)))),

        // underpromotion
        (Some(67 * 64 + Square::G7.to_index()), Some(ChessMove::new(Square::G2, Square::G1, Some(Piece::Rook)))),
        (Some(70 * 64 + Square::G7.to_index()), Some(ChessMove::new(Square::G2, Square::H1, Some(Piece::Rook)))),
        (Some(64 * 64 + Square::G7.to_index()), Some(ChessMove::new(Square::G2, Square::F1, Some(Piece::Rook)))),
        (Some(68 * 64 + Square::G7.to_index()), Some(ChessMove::new(Square::G2, Square::G1, Some(Piece::Bishop)))),
        (Some(71 * 64 + Square::G7.to_index()), Some(ChessMove::new(Square::G2, Square::H1, Some(Piece::Bishop)))),
        (Some(65 * 64 + Square::G7.to_index()), Some(ChessMove::new(Square::G2, Square::F1, Some(Piece::Bishop)))),
        (Some(69 * 64 + Square::G7.to_index()), Some(ChessMove::new(Square::G2, Square::G1, Some(Piece::Knight)))),
        (Some(72 * 64 + Square::G7.to_index()), Some(ChessMove::new(Square::G2, Square::H1, Some(Piece::Knight)))),
        (Some(66 * 64 + Square::G7.to_index()), Some(ChessMove::new(Square::G2, Square::F1, Some(Piece::Knight)))),
    ])
}

#[test]
fn en_passant() {
    let white_board = board("8/8/5k2/1pP5/8/5K2/8/8 w - b6 0 2");
    let black_board = board("8/8/5k2/8/1pP5/5K2/8/8 b - c3 0 1");

    test_policy_pairs(&white_board, &[
        (Some((7 * 7 + 0) * 64 + Square::C5.to_index()), Some(ChessMove::new(Square::C5, Square::B6, None)))
    ]);

    // careful, move indices are from the POV of black!
    test_policy_pairs(&black_board, &[
        (Some((1 * 7 + 0) * 64 + Square::B5.to_index()), Some(ChessMove::new(Square::B4, Square::C3, None)))
    ]);
}

#[test]
fn castles() {
    // both players can castle both ways immediately
    let white_board = board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    let black_board = board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");

    test_policy_pairs(&white_board, &[
        (Some((2 * 7 + 1) * 64 + Square::E1.to_index()), Some(ChessMove::new(Square::E1, Square::G1, None))),
        (Some((6 * 7 + 1) * 64 + Square::E1.to_index()), Some(ChessMove::new(Square::E1, Square::C1, None))),
    ]);

    // careful, move indices are from the POV of black!
    test_policy_pairs(&black_board, &[
        (Some((2 * 7 + 1) * 64 + Square::E1.to_index()), Some(ChessMove::new(Square::E8, Square::G8, None))),
        (Some((6 * 7 + 1) * 64 + Square::E1.to_index()), Some(ChessMove::new(Square::E8, Square::C8, None))),
    ]);
}

fn board(fen: &str) -> ChessBoard {
    ChessBoard::new(chess::Board::from_str(fen).unwrap(), Rules::default())
}

pub fn test_policy_pairs(board: &ChessBoard, pairs: &[(Option<usize>, Option<ChessMove>)]) {
    let mapper = ChessStdMapper;

    test_valid_mapping(mapper, board);

    println!("Running on board\n  {}", board);
    println!("Using mapper {:?}", mapper);

    for &(index, mv) in pairs {
        println!("  Testing pair {:?} <-> {}", index, display_option(mv));

        if let Some(mv) = mv {
            assert!(board.is_available_move(mv), "Move is not available on current board");

            println!("    mv -> index");
            let classified = ClassifiedPovMove::from_move(mv);
            println!("    {:?} -> {}", classified, classified.to_channel());

            assert_eq!(
                index, mapper.move_to_index(board, mv),
                "Wrong index for move {}", mv
            );
        }

        if let Some(index) = index {
            println!("    index -> mv");

            let channel = index / 64;
            let classified = ClassifiedPovMove::from_channel(channel);
            println!("    {} -> {:?}", channel, classified);

            let returned_move = mapper.index_to_move(board, index);
            assert_eq!(
                mv, returned_move,
                "Expected move {}, got {} for index {}",
                display_option(mv), display_option(returned_move), index,
            );
        }

        println!();
    }

    println!();
}

