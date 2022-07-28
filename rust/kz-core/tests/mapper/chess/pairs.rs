use board_game::board::Board;
use board_game::chess::{ChessMove, Piece, Square};
use board_game::games::chess::{ChessBoard, Rules};

use kz_core::mapping::chess::{ChessHistoryMapper, ChessLegacyConvPolicyMapper, ChessStdMapper, ClassifiedPovMove};
use kz_core::mapping::PolicyMapper;
use kz_util::display::display_option;

use crate::mapper::{test_valid_mapping, test_valid_policy_mapping};

#[test]
fn basic_board_mapping() {
    test_valid_mapping(ChessStdMapper, &ChessBoard::default());
}

#[test]
fn queen_distance_white() {
    // mostly empty with white queen on A1
    let board = board("8/8/8/6k1/8/6K1/8/Q7 w - - 0 1");

    test_pairs(
        &board,
        &[
            (0 * 64, Some(ChessMove::new(Square::A1, Square::A2, None))),
            (1 * 64, Some(ChessMove::new(Square::A1, Square::A3, None))),
            (2 * 64, Some(ChessMove::new(Square::A1, Square::A4, None))),
            (3 * 64, Some(ChessMove::new(Square::A1, Square::A5, None))),
            (4 * 64, Some(ChessMove::new(Square::A1, Square::A6, None))),
            (5 * 64, Some(ChessMove::new(Square::A1, Square::A7, None))),
            (6 * 64, Some(ChessMove::new(Square::A1, Square::A8, None))),
        ],
    );
}

#[test]
fn queen_distance_black() {
    // mostly empty with black queen on A8
    let board = board("q7/8/8/6k1/8/6K1/8/8 b - - 0 1");

    test_pairs(
        &board,
        &[
            (0 * 64, Some(ChessMove::new(Square::A8, Square::A7, None))),
            (1 * 64, Some(ChessMove::new(Square::A8, Square::A6, None))),
            (2 * 64, Some(ChessMove::new(Square::A8, Square::A5, None))),
            (3 * 64, Some(ChessMove::new(Square::A8, Square::A4, None))),
            (4 * 64, Some(ChessMove::new(Square::A8, Square::A3, None))),
            (5 * 64, Some(ChessMove::new(Square::A8, Square::A2, None))),
            (6 * 64, Some(ChessMove::new(Square::A8, Square::A1, None))),
        ],
    );
}

#[test]
fn queen_direction_white() {
    // mostly empty with white queen on D4
    let board = board("8/8/6k1/8/3Q4/6K1/8/8 w - - 0 1");

    let d4 = Square::D4.to_index();
    test_pairs(
        &board,
        &[
            ((0 * 7 * 64 + d4), Some(ChessMove::new(Square::D4, Square::D5, None))),
            ((1 * 7 * 64 + d4), Some(ChessMove::new(Square::D4, Square::E5, None))),
            ((2 * 7 * 64 + d4), Some(ChessMove::new(Square::D4, Square::E4, None))),
            ((3 * 7 * 64 + d4), Some(ChessMove::new(Square::D4, Square::E3, None))),
            ((4 * 7 * 64 + d4), Some(ChessMove::new(Square::D4, Square::D3, None))),
            ((5 * 7 * 64 + d4), Some(ChessMove::new(Square::D4, Square::C3, None))),
            ((6 * 7 * 64 + d4), Some(ChessMove::new(Square::D4, Square::C4, None))),
            ((7 * 7 * 64 + d4), Some(ChessMove::new(Square::D4, Square::C5, None))),
        ],
    )
}

#[test]
fn queen_direction_black() {
    // mostly empty with black queen on D5
    let board = board("8/8/6k1/3q4/8/6K1/8/8 b - - 0 1");

    let d4 = Square::D4.to_index();
    test_pairs(
        &board,
        &[
            ((0 * 7 * 64 + d4), Some(ChessMove::new(Square::D5, Square::D4, None))),
            ((1 * 7 * 64 + d4), Some(ChessMove::new(Square::D5, Square::E4, None))),
            ((2 * 7 * 64 + d4), Some(ChessMove::new(Square::D5, Square::E5, None))),
            ((3 * 7 * 64 + d4), Some(ChessMove::new(Square::D5, Square::E6, None))),
            ((4 * 7 * 64 + d4), Some(ChessMove::new(Square::D5, Square::D6, None))),
            ((5 * 7 * 64 + d4), Some(ChessMove::new(Square::D5, Square::C6, None))),
            ((6 * 7 * 64 + d4), Some(ChessMove::new(Square::D5, Square::C5, None))),
            ((7 * 7 * 64 + d4), Some(ChessMove::new(Square::D5, Square::C4, None))),
        ],
    )
}

#[test]
fn knight_direction_white() {
    // mostly empty with white knight on D4
    let board = board("8/8/6k1/8/3N4/6K1/8/8 w - - 0 1");

    let d4 = Square::D4.to_index();
    test_pairs(
        &board,
        &[
            (56 * 64 + d4, Some(ChessMove::new(Square::D4, Square::E6, None))),
            (57 * 64 + d4, Some(ChessMove::new(Square::D4, Square::F5, None))),
            (58 * 64 + d4, Some(ChessMove::new(Square::D4, Square::F3, None))),
            (59 * 64 + d4, Some(ChessMove::new(Square::D4, Square::E2, None))),
            (60 * 64 + d4, Some(ChessMove::new(Square::D4, Square::C2, None))),
            (61 * 64 + d4, Some(ChessMove::new(Square::D4, Square::B3, None))),
            (62 * 64 + d4, Some(ChessMove::new(Square::D4, Square::B5, None))),
            (63 * 64 + d4, Some(ChessMove::new(Square::D4, Square::C6, None))),
        ],
    )
}

#[test]
fn knight_direction_black() {
    // mostly empty with black knight on D5
    let board = board("8/8/6k1/3n4/8/6K1/8/8 b - - 0 1");

    let d4 = Square::D4.to_index();
    test_pairs(
        &board,
        &[
            (56 * 64 + d4, Some(ChessMove::new(Square::D5, Square::E3, None))),
            (57 * 64 + d4, Some(ChessMove::new(Square::D5, Square::F4, None))),
            (58 * 64 + d4, Some(ChessMove::new(Square::D5, Square::F6, None))),
            (59 * 64 + d4, Some(ChessMove::new(Square::D5, Square::E7, None))),
            (60 * 64 + d4, Some(ChessMove::new(Square::D5, Square::C7, None))),
            (61 * 64 + d4, Some(ChessMove::new(Square::D5, Square::B6, None))),
            (62 * 64 + d4, Some(ChessMove::new(Square::D5, Square::B4, None))),
            (63 * 64 + d4, Some(ChessMove::new(Square::D5, Square::C3, None))),
        ],
    )
}

#[test]
fn white_potential_promotions() {
    // lots of promotion opportunities for white
    let board = board("r1r5/1P4R1/5RNP/2k5/5K2/pnr5/1r4p1/5R1R w - - 0 1");

    test_pairs(
        &board,
        &[
            // rook, no promotion
            (
                ((0 * 7 + 1) * 64 + Square::F6.to_index()),
                Some(ChessMove::new(Square::F6, Square::F8, None)),
            ),
            (
                ((0 * 7 + 0) * 64 + Square::G7.to_index()),
                Some(ChessMove::new(Square::G7, Square::G8, None)),
            ),
            // knight, no promotion
            (
                (63 * 64 + Square::G6.to_index()),
                Some(ChessMove::new(Square::G6, Square::F8, None)),
            ),
            (
                (56 * 64 + Square::G6.to_index()),
                Some(ChessMove::new(Square::G6, Square::H8, None)),
            ),
            // promotion to queen
            (
                ((7 * 7 + 0) * 64 + Square::B7.to_index()),
                Some(ChessMove::new(Square::B7, Square::A8, Some(Piece::Queen))),
            ),
            (
                ((0 * 7 + 0) * 64 + Square::B7.to_index()),
                Some(ChessMove::new(Square::B7, Square::B8, Some(Piece::Queen))),
            ),
            (
                ((1 * 7 + 0) * 64 + Square::B7.to_index()),
                Some(ChessMove::new(Square::B7, Square::C8, Some(Piece::Queen))),
            ),
            // underpromotion
            (
                (64 * 64 + Square::B7.to_index()),
                Some(ChessMove::new(Square::B7, Square::A8, Some(Piece::Rook))),
            ),
            (
                (67 * 64 + Square::B7.to_index()),
                Some(ChessMove::new(Square::B7, Square::B8, Some(Piece::Rook))),
            ),
            (
                (70 * 64 + Square::B7.to_index()),
                Some(ChessMove::new(Square::B7, Square::C8, Some(Piece::Rook))),
            ),
            (
                (65 * 64 + Square::B7.to_index()),
                Some(ChessMove::new(Square::B7, Square::A8, Some(Piece::Bishop))),
            ),
            (
                (68 * 64 + Square::B7.to_index()),
                Some(ChessMove::new(Square::B7, Square::B8, Some(Piece::Bishop))),
            ),
            (
                (71 * 64 + Square::B7.to_index()),
                Some(ChessMove::new(Square::B7, Square::C8, Some(Piece::Bishop))),
            ),
            (
                (66 * 64 + Square::B7.to_index()),
                Some(ChessMove::new(Square::B7, Square::A8, Some(Piece::Knight))),
            ),
            (
                (69 * 64 + Square::B7.to_index()),
                Some(ChessMove::new(Square::B7, Square::B8, Some(Piece::Knight))),
            ),
            (
                (72 * 64 + Square::B7.to_index()),
                Some(ChessMove::new(Square::B7, Square::C8, Some(Piece::Knight))),
            ),
        ],
    )
}

#[test]
fn black_potential_promotions() {
    // lots of promotion opportunities for black
    let board = board("r1r5/1P4R1/5RNP/2k5/5K2/pnr5/1r4p1/5R1R b - - 0 1");

    // careful, move indices are from the POV of black!
    test_pairs(
        &board,
        &[
            // rook, no promotion
            (
                ((0 * 7 + 1) * 64 + Square::C6.to_index()),
                Some(ChessMove::new(Square::C3, Square::C1, None)),
            ),
            (
                ((0 * 7 + 0) * 64 + Square::B7.to_index()),
                Some(ChessMove::new(Square::B2, Square::B1, None)),
            ),
            // knight, no promotion
            (
                (56 * 64 + Square::B6.to_index()),
                Some(ChessMove::new(Square::B3, Square::C1, None)),
            ),
            (
                (63 * 64 + Square::B6.to_index()),
                Some(ChessMove::new(Square::B3, Square::A1, None)),
            ),
            // promotion to queen
            (
                ((7 * 7 + 0) * 64 + Square::G7.to_index()),
                Some(ChessMove::new(Square::G2, Square::F1, Some(Piece::Queen))),
            ),
            (
                ((0 * 7 + 0) * 64 + Square::G7.to_index()),
                Some(ChessMove::new(Square::G2, Square::G1, Some(Piece::Queen))),
            ),
            (
                ((1 * 7 + 0) * 64 + Square::G7.to_index()),
                Some(ChessMove::new(Square::G2, Square::H1, Some(Piece::Queen))),
            ),
            // underpromotion
            (
                (67 * 64 + Square::G7.to_index()),
                Some(ChessMove::new(Square::G2, Square::G1, Some(Piece::Rook))),
            ),
            (
                (70 * 64 + Square::G7.to_index()),
                Some(ChessMove::new(Square::G2, Square::H1, Some(Piece::Rook))),
            ),
            (
                (64 * 64 + Square::G7.to_index()),
                Some(ChessMove::new(Square::G2, Square::F1, Some(Piece::Rook))),
            ),
            (
                (68 * 64 + Square::G7.to_index()),
                Some(ChessMove::new(Square::G2, Square::G1, Some(Piece::Bishop))),
            ),
            (
                (71 * 64 + Square::G7.to_index()),
                Some(ChessMove::new(Square::G2, Square::H1, Some(Piece::Bishop))),
            ),
            (
                (65 * 64 + Square::G7.to_index()),
                Some(ChessMove::new(Square::G2, Square::F1, Some(Piece::Bishop))),
            ),
            (
                (69 * 64 + Square::G7.to_index()),
                Some(ChessMove::new(Square::G2, Square::G1, Some(Piece::Knight))),
            ),
            (
                (72 * 64 + Square::G7.to_index()),
                Some(ChessMove::new(Square::G2, Square::H1, Some(Piece::Knight))),
            ),
            (
                (66 * 64 + Square::G7.to_index()),
                Some(ChessMove::new(Square::G2, Square::F1, Some(Piece::Knight))),
            ),
        ],
    )
}

#[test]
fn en_passant() {
    let white_board = board("8/8/5k2/1pP5/8/5K2/8/8 w - b6 0 2");
    let black_board = board("8/8/5k2/8/1pP5/5K2/8/8 b - c3 0 1");

    test_pairs(
        &white_board,
        &[(
            ((7 * 7 + 0) * 64 + Square::C5.to_index()),
            Some(ChessMove::new(Square::C5, Square::B6, None)),
        )],
    );

    // careful, move indices are from the POV of black!
    test_pairs(
        &black_board,
        &[(
            ((1 * 7 + 0) * 64 + Square::B5.to_index()),
            Some(ChessMove::new(Square::B4, Square::C3, None)),
        )],
    );
}

#[test]
fn castles() {
    // both players can castle both ways immediately
    let white_board = board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    let black_board = board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");

    test_pairs(
        &white_board,
        &[
            (
                ((2 * 7 + 1) * 64 + Square::E1.to_index()),
                Some(ChessMove::new(Square::E1, Square::G1, None)),
            ),
            (
                ((6 * 7 + 1) * 64 + Square::E1.to_index()),
                Some(ChessMove::new(Square::E1, Square::C1, None)),
            ),
        ],
    );

    // careful, move indices are from the POV of black!
    test_pairs(
        &black_board,
        &[
            (
                ((2 * 7 + 1) * 64 + Square::E1.to_index()),
                Some(ChessMove::new(Square::E8, Square::G8, None)),
            ),
            (
                ((6 * 7 + 1) * 64 + Square::E1.to_index()),
                Some(ChessMove::new(Square::E8, Square::C8, None)),
            ),
        ],
    );
}

fn board(fen: &str) -> ChessBoard {
    ChessBoard::new_without_history_fen(fen, Rules::default())
}

fn test_pairs(board: &ChessBoard, pairs: &[(usize, Option<ChessMove>)]) {
    // test other mapper with a variety of chess boards
    test_valid_mapping(ChessStdMapper, board);
    test_valid_policy_mapping(ChessLegacyConvPolicyMapper, board);
    for length in [0, 1, 8] {
        test_valid_mapping(ChessHistoryMapper::new(length), board);
    }

    // do the actual conv mapper test
    let mapper = ChessLegacyConvPolicyMapper;

    println!("Running on board\n  {}", board);
    println!("Using mapper {:?}", mapper);

    for &(index, mv) in pairs {
        println!("  Testing pair {:?} <-> {}", index, display_option(mv));

        if let Some(mv) = mv {
            assert!(board.is_available_move(mv), "Move is not available on current board");

            println!("    mv -> index");
            let classified = ClassifiedPovMove::from_move(mv);
            println!("    {:?} -> {}", classified, classified.to_channel());

            assert_eq!(index, mapper.move_to_index(board, mv), "Wrong index for move {}", mv);
        }

        println!("    index -> mv");

        let channel = index / 64;
        let classified = ClassifiedPovMove::from_channel(channel);
        println!("    {} -> {:?}", channel, classified);

        let returned_move = mapper.index_to_move(board, index);
        assert_eq!(
            mv,
            returned_move,
            "Expected move {}, got {} for index {}",
            display_option(mv),
            display_option(returned_move),
            index,
        );

        println!();
    }

    println!();
}
