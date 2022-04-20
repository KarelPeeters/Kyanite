use std::panic::resume_unwind;

use board_game::board::Board;
use internal_iterator::InternalIterator;

use kz_core::mapping::bit_buffer::BitBuffer;
use kz_core::mapping::{BoardMapper, InputMapper, PolicyMapper};
use kz_util::display_option;

mod ataxx;
mod chess_flat_gen;
mod chess_manual_conv;
mod chess_random;

pub fn test_valid_mapping<B: Board, M: BoardMapper<B>>(mapper: M, board: &B) {
    if !board.is_done() {
        test_valid_policy_mapping(mapper, board);
    }

    test_valid_input_mapping(mapper, board);
}

pub fn test_valid_input_mapping<B: Board, M: InputMapper<B>>(mapper: M, board: &B) {
    let mut bools = BitBuffer::new(mapper.input_bool_len());
    let mut scalars = vec![];
    mapper.encode_input(&mut bools, &mut scalars, board);
    assert_eq!(bools.len(), mapper.input_bool_len());
    assert_eq!(scalars.len(), mapper.input_scalar_count());
}

pub fn test_valid_policy_mapping<B: Board, M: PolicyMapper<B>>(mapper: M, board: &B) {
    println!("Testing policy for {:?} on {:?}", mapper, board);

    assert!(!board.is_done());
    test_move_to_index(mapper, board);
    test_index_to_move(mapper, board);
}

pub fn test_move_to_index<B: Board, M: PolicyMapper<B>>(mapper: M, board: &B) {
    let mut prev = vec![vec![]; mapper.policy_len()];
    board.available_moves().for_each(|mv: B::Move| {
        match mapper.move_to_index(board, mv) {
            None => assert_eq!(1, board.available_moves().count()),
            Some(index) => {
                //check that roundtrip works out
                let remapped_move = mapper.index_to_move(board, index);
                assert_eq!(
                    Some(mv),
                    remapped_move,
                    "Failed move roundtrip: {} -> {} -> {}",
                    mv,
                    index,
                    display_option(remapped_move)
                );

                // keep the move so we can report duplicate indices later
                prev[index].push(mv);
            }
        }
    });

    // now we can easily check for duplicate moves and report all of them
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

pub fn test_index_to_move<B: Board, M: PolicyMapper<B>>(mapper: M, board: &B) {
    for index in 0..mapper.policy_len() {
        let mv = std::panic::catch_unwind(|| mapper.index_to_move(board, index));

        match mv {
            Err(e) => {
                eprintln!("Panic while mapping index {} to move on board\n  {}", index, board);
                resume_unwind(e);
            }
            Ok(mv) => {
                if let Some(mv) = mv {
                    let available = std::panic::catch_unwind(|| board.is_available_move(mv));
                    match available {
                        Ok(_) => {}
                        Err(e) => {
                            eprintln!(
                                "Panic while using move {} from index {} to move on board\n  {}",
                                mv, index, board
                            );
                            resume_unwind(e);
                        }
                    }
                }
            }
        }
    }
}
