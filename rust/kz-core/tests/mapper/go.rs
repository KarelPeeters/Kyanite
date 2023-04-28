use crate::mapper::test_valid_mapping;
use board_game::games::go::{GoBoard, Rules};
use kz_core::mapping::go::GoStdMapper;

#[test]
fn go_empty() {
    let board = GoBoard::new(9, 0, Rules::tromp_taylor());
    let mapper = GoStdMapper::new(board.size());
    test_valid_mapping(mapper, &board);
}
