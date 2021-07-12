use rand::Rng;

use crate::ai::solver::{find_forcing_winner, is_double_forced_draw};
use crate::board::{Board, Outcome};

/// Generate a `Board` by playing `n` random moves on `start`.
pub fn random_board_with_moves<B: Board>(start: &B, n: u32, rng: &mut impl Rng) -> B {
    //this implementation could be made faster with backtracking instead of starting from scratch,
    // but this only starts to matter for very high n and that's not really the main use case

    loop {
        let mut board = start.clone();
        for _ in 0..n {
            if board.is_done() { break; }
            board.play(board.random_available_move(rng))
        }
        return board;
    }
}

pub fn random_board_with_outcome<B: Board>(start: &B, outcome: Outcome, rng: &mut impl Rng) -> B {
    loop {
        let mut board = start.clone();
        loop {
            if let Some(actual) = board.outcome() {
                if actual == outcome { return board; }
                break;
            }

            board.play(board.random_available_move(rng))
        }
    }
}

/// Generate a `Board` by playing random moves until a forced win in `depth` moves is found for `start.next_player`.
pub fn random_board_with_forced_win<B: Board>(start: &B, depth: u32, rng: &mut impl Rng) -> B {
    if !B::can_lose_after_move() {
        assert!(depth % 2 == 1, "forced win in an even number of moves is impossible \
                                (because the last move would be by the opponent)");
    }

    random_board_with_depth_condition(start, depth, rng, |board, depth| {
        find_forcing_winner(board, depth) == Some(board.next_player())
    })
}

/// Generate a random board with a *double forced draw* in `depth` moves, meaning that no matter what either player does
/// it's impossible for someone to win.
pub fn random_board_with_double_forced_draw<B: Board>(start: &B, depth: u32, rng: &mut impl Rng) -> B {
    random_board_with_depth_condition(start, depth, rng, |board, depth| {
        is_double_forced_draw(board, depth).unwrap_or(false)
    })
}

/// Generate a random board such that `cond(board, depth) & !cond(board, depth-1)`.
fn random_board_with_depth_condition<B: Board>(start: &B, depth: u32, rng: &mut impl Rng, cond: impl Fn(&B, u32) -> bool) -> B {
    loop {
        let mut board = start.clone();

        loop {
            let deep_match = cond(&board, depth);
            if deep_match {
                let shallow_match = depth > 0 && cond(&board, depth - 1);
                if shallow_match { break; }

                return board;
            }

            if board.is_done() { break; }
            board.play(board.random_available_move(rng));
        }
    }
}
