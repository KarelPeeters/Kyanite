use buffered_reader::Memory;

use pgn_reader::PgnReader;

fn read(input: &str) -> PgnReader<Memory<()>> {
    let read = Memory::new(input.as_bytes());
    PgnReader::new(read)
}

#[test]
fn empty() {
    assert_eq!(read("").next().unwrap(), None);
}

#[test]
fn single() {
    let mut reader = read("[Test \"test\"]\n1. e4\n");

    let game = reader.next().unwrap().unwrap();
    assert_eq!(game.header, "[Test \"test\"]\n");
    assert_eq!(game.moves, "1. e4\n");
    assert_eq!(game.header("Test"), Some("test"));

    assert_eq!(reader.next().unwrap(), None);
}