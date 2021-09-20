use board_game::board::Board;
use internal_iterator::InternalIterator;

use alpha_zero::mapping::PolicyMapper;
use alpha_zero::util::display_option;

mod chess_manual;
mod chess_random;

pub fn test_valid_mapping<B: Board, M: PolicyMapper<B>>(mapper: M, board: &B) {
    assert!(!board.is_done());

    let mut prev = vec![vec![]; M::POLICY_SIZE];
    board.available_moves().for_each(|mv: B::Move| {
        match mapper.move_to_index(board, mv) {
            None => assert_eq!(1, board.available_moves().count()),
            Some(index) => {
                let remapped_move = mapper.index_to_move(board, index);
                assert_eq!(
                    Some(mv), remapped_move,
                    "Failed move roundtrip: {} -> {} -> {}",
                    mv, index, display_option(remapped_move)
                );

                prev[index].push(mv);
            }
        }
    });

    let mut err = false;
    for (i, moves) in prev.iter().enumerate() {
        if moves.len() > 1 {
            err = true;
            eprintln!("  Multiple moves mapped to index {}:", i);
            for mv in moves {
                eprintln!("    {}", mv);
            }
        }
    }

    println!("  On board {}", board);
    assert!(!err, "Multiple moves mapped to the same index");
}