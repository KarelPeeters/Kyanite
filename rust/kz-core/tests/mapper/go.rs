use crate::mapper::test_valid_mapping;
use board_game::games::go::{GoBoard, Komi, Rules};
use kz_core::mapping::go::GoStdMapper;

#[test]
fn go_empty_basic() {
    let board = GoBoard::new(9, Komi::zero(), Rules::tromp_taylor());
    let mapper = GoStdMapper::new(board.size(), false);
    test_valid_mapping(mapper, &board);
}

#[test]
fn go_empty_territory() {
    let board = GoBoard::new(9, Komi::zero(), Rules::tromp_taylor());
    let mapper = GoStdMapper::new(board.size(), true);
    test_valid_mapping(mapper, &board);
}
