use board_game::games::ataxx::AtaxxBoard;

use alpha_zero::mapping::ataxx::AtaxxStdMapper;

use crate::mapper::test_valid_mapping;

#[test]
fn sizes_start() {
    for size in 3..AtaxxBoard::MAX_SIZE {
        test_valid_mapping(AtaxxStdMapper::new(size), &AtaxxBoard::diagonal(size))
    }
}

#[test]
fn forced_pass_5() {
    let board = AtaxxBoard::from_fen("xoooo/xoo2/ooooo/ooooo/xxooo x 0 0").unwrap();
    test_valid_mapping(AtaxxStdMapper::new(5), &board);
}