use buffered_reader::Memory;

use pgn_reader::{PgnOutcome, PgnReader};

fn read(input: &str) -> PgnReader<Memory<()>> {
    let read = Memory::new(input.as_bytes());
    PgnReader::new(read)
}

#[test]
fn empty() {
    assert_eq!(read("").next().unwrap(), None);
}

#[test]
fn short() {
    let mut reader = read("[Test \"test\"]\n1. e4 1/2-1/2\n");

    let game = reader.next().unwrap().unwrap();
    assert_eq!(game.header("Test"), Some("test"));
    assert_eq!(game.parse_moves(), (vec!["e4"], PgnOutcome::Draw));

    assert_eq!(reader.next().unwrap(), None);
}

#[test]
fn multi_digit() {
    let mut reader = read("[Test \"test\"]\n10. e4 1/2-1/2\n");

    let game = reader.next().unwrap().unwrap();
    assert_eq!(game.header("Test"), Some("test"));
    assert_eq!(game.parse_moves(), (vec!["e4"], PgnOutcome::Draw));

    assert_eq!(reader.next().unwrap(), None);
}

#[test]
fn multiple_moves_per_turn() {
    let mut reader = read("[Test \"test\"]\n1. e4 e5 2. d4 d5 1-0\n");

    let game = reader.next().unwrap().unwrap();
    assert_eq!(game.header("Test"), Some("test"));
    assert_eq!(game.parse_moves(), (vec!["e4", "e5", "d4", "d5"], PgnOutcome::WinWhite));

    assert_eq!(reader.next().unwrap(), None);
}

#[test]
fn variation() {
    let mut reader = read("[Test \"test\"]\n1. e4 { d5 6d d8 } 1... d5 { [%clk 0:02:55] } 1-0\n");

    let game = reader.next().unwrap().unwrap();
    assert_eq!(game.header("Test"), Some("test"));
    assert_eq!(game.parse_moves(), (vec!["e4", "d5"], PgnOutcome::WinWhite));

    assert_eq!(reader.next().unwrap(), None);
}