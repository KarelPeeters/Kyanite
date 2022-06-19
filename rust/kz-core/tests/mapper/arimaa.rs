use std::str::FromStr;

use board_game::board::{Board, BoardMoves, Player};
use board_game::games::arimaa::ArimaaBoard;
use internal_iterator::InternalIterator;

use kz_core::mapping::arimaa::ArimaaSplitMapper;

use crate::mapper::test_valid_mapping;

#[test]
fn empty() {
    let board = ArimaaBoard::default();
    test_valid_mapping(ArimaaSplitMapper, &board);
}

#[test]
fn typical_placement() {
    let board = ArimaaBoard::from_str(BASIC_SETUP).unwrap();
    test_valid_mapping(ArimaaSplitMapper, &board);
}

#[test]
fn can_pass() {
    let mut board = ArimaaBoard::from_str(BASIC_SETUP).unwrap();
    board.play(board.available_moves().next().unwrap());

    test_valid_mapping(ArimaaSplitMapper, &board);
}

#[test]
fn gold_goal() {
    let board = ArimaaBoard::from_str(GOLD_GOAL).unwrap();
    test_valid_mapping(ArimaaSplitMapper, &board);
}

const BASIC_SETUP: &str = "
     +-----------------+
    8| r r r r r r r r |
    7| d h c e m c h d |
    6| . . x . . x . . |
    5| . . . . . . . . |
    4| . . . . . . . . |
    3| . . x . . x . . |
    2| D H C M E C H D |
    1| R R R R R R R R |
     +-----------------+
       a b c d e f g h  
";

const GOLD_GOAL: &str = "
    23w
     +-----------------+
    8| r R r r   r r r |
    7|     d           |
    6|   D X c   X     |
    5|         R m     |
    4|                 |
    3|     X     X     |
    2|           d     |
    1| R   R R R R     |
     +-----------------+
       a b c d e f g h
";
