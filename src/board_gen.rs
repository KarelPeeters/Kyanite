use rand::Rng;
use crate::board::Board;
use crate::minimax::evaluate_minimax;

/// Generate a `Board` by playing `n` random moves.
pub fn random_board_with_moves(n: u32, rng: &mut impl Rng) -> Board {
    //this implementation could be made faster with backtracking instead of starting from scratch,
    // but this only starts to matter for very high n and that's not really the main use case

    assert!(n <= 81, "cannot generate board with '{}' > 81 moves", n);

    loop {
        let mut board = Board::new();
        for _ in 0..n {
            match board.random_available_move(rng) {
                Some(mv) => board.play(mv),

                //board is done, so we can't play a move. discard and start with a new board
                None => break,
            };
        }
        return board;
    }
}

/// Generate a `Board` by playing random moves until a forced win in `depth` moves is found for
/// `board.next_player` by minimax.
pub fn random_board_with_forced_win(depth: u32, rng: &mut impl Rng) -> Board {
    assert!(depth <= 81, "cannot generate board with forced win in '{}' > 81 moves", depth);
    assert!(depth % 2 == 1, "forced win in an even number of moves is impossible \
                                (because the last move would be by the opponent)");

    // we don't need to check the first few moves since it's not possible to win immediately
    const FASTEST_POSSIBLE_WIN: u32 = 9 + 8;
    let certain_moves = FASTEST_POSSIBLE_WIN.saturating_sub(depth);

    loop {
        let mut board = random_board_with_moves(certain_moves, rng);

        loop {
            let deep_eval = evaluate_minimax(&board, depth);

            if deep_eval.is_forced_win() {
                let shallow_win = depth > 1 && evaluate_minimax(&board, depth - 1).is_forced_win();

                if !shallow_win {
                    return board;
                } else {
                    break
                }
            }

            match board.random_available_move(rng) {
                None => {
                    break;
                }
                Some(mv) => {
                    board.play(mv);
                }
            }
        }
    }
}
