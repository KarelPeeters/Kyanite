use buffered_reader::Memory;
use internal_iterator::InternalIterator;

use pgn_reader::{PgnMove, PgnReader, PgnResult};

fn read(input: &str) -> PgnReader<Memory<()>> {
    let read = Memory::new(input.as_bytes());
    PgnReader::new(read)
}

#[test]
fn empty() {
    assert_eq!(read("").next_game().unwrap(), None);
}

#[test]
fn short() {
    let mut reader = read("[Test \"test\"]\n1. e4 1/2-1/2\n");

    let game = reader.next_game().unwrap().unwrap();
    assert_eq!(game.header("Test"), Some("test"));
    assert_eq!(game.move_iter().collect::<Vec<_>>(), vec![PgnMove { mv: "e4", comment: None }]);
    assert_eq!(game.result(), PgnResult::Draw);

    assert_eq!(reader.next_game().unwrap(), None);
}

#[test]
fn multi_digit() {
    let mut reader = read("[Test \"test\"]\n10. e4 1/2-1/2\n");

    let game = reader.next_game().unwrap().unwrap();
    assert_eq!(game.header("Test"), Some("test"));
    assert_eq!(game.move_iter().collect::<Vec<_>>(), vec![PgnMove { mv: "e4", comment: None }]);
    assert_eq!(game.result(), PgnResult::Draw);

    assert_eq!(reader.next_game().unwrap(), None);
}

#[test]
fn multiple_moves_per_turn() {
    let mut reader = read("[Test \"test\"]\n1. e4 e5 2. d4 d5 1-0\n");

    let game = reader.next_game().unwrap().unwrap();
    assert_eq!(game.header("Test"), Some("test"));
    assert_eq!(
        game.move_iter().collect::<Vec<_>>(),
        vec![
            PgnMove { mv: "e4", comment: None },
            PgnMove { mv: "e5", comment: None },
            PgnMove { mv: "d4", comment: None },
            PgnMove { mv: "d5", comment: None },
        ]
    );
    assert_eq!(game.result(), PgnResult::WinWhite);

    assert_eq!(reader.next_game().unwrap(), None);
}

#[test]
fn variation() {
    let mut reader = read("[Test \"test\"]\n1. e4 { d5 6d d8 } 1... d5 { [%clk 0:02:55] } { foo {  dsf  } { 0-1 } } a6 1-0\n");

    let game = reader.next_game().unwrap().unwrap();
    assert_eq!(game.header("Test"), Some("test"));
    assert_eq!(
        game.move_iter().collect::<Vec<_>>(),
        vec![
            PgnMove { mv: "e4", comment: Some("d5 6d d8") },
            PgnMove { mv: "d5", comment: Some("[%clk 0:02:55]") },
            PgnMove { mv: "a6", comment: None },
        ]
    );
    assert_eq!(game.result(), PgnResult::WinWhite);

    assert_eq!(reader.next_game().unwrap(), None);
}

#[test]
fn comments() {
    let mv = PgnMove {
        mv: "e4",
        comment: Some("[%eval 0.16] [%clk 0:00:30]"),
    };

    assert_eq!(mv.field("eval"), Some("0.16"));
    assert_eq!(mv.field("clk"), Some("0:00:30"));
    assert_eq!(mv.field("derp"), None);
}