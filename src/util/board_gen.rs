use rand::Rng;

use crate::ai::solver::find_forcing_winner;
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

/// Generate a `Board` by playing random moves until a forced win in `depth` moves is found for
/// `board.next_player` by minimax.
pub fn random_board_with_forced_win<B: Board>(start: &B, depth: u32, rng: &mut impl Rng) -> B {
    if !B::can_lose_after_move() {
        assert!(depth % 2 == 1, "forced win in an even number of moves is impossible \
                                (because the last move would be by the opponent)");
    }

    loop {
        let mut board = start.clone();

        loop {
            let deep_eval = find_forcing_winner(&board, depth);

            if deep_eval.is_some() {
                let shallow_win = depth > 1 && find_forcing_winner(&board, depth - 1).is_some();
                if shallow_win { break; }
                return board;
            }

            if board.is_done() { break; }
            board.play(board.random_available_move(rng));
        }
    }
}

fn is_double_forced_draw(board: &impl Board, depth: u32) -> bool {
    if board.outcome() == Some(Outcome::Draw) { return true; }
    if board.outcome().is_some() || depth == 0 { return false; }

    board.available_moves()
        .all(|mv| is_double_forced_draw(&board.clone_and_play(mv), depth - 1))
}

/// Generate a random board with a *double forced draw* in `depth` moves, meaning that no matter what either player does
/// it's impossible for someone to win.
pub fn random_board_with_double_forced_draw<B: Board>(start: &B, depth: u32, rng: &mut impl Rng) -> B {
    loop {
        let mut board = start.clone();

        loop {
            if is_double_forced_draw(&board, depth) {
                let shallow_draw = depth > 0 && is_double_forced_draw(&board, depth - 1);
                if shallow_draw { break; }
                return board;
            };

            if board.is_done() { break; }
            board.play(board.random_available_move(rng));
        }
    }
}
