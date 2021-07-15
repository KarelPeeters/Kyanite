use std::collections::{BTreeMap, HashSet};
use std::collections::hash_map::RandomState;
use std::iter::FromIterator;

use internal_iterator::InternalIterator;
use itertools::Itertools;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoroshiro64StarStar;

use sttt::board::{Board, Outcome, Player};
use sttt::games::ataxx::AtaxxBoard;
use sttt::games::sttt::STTTBoard;
use sttt::symmetry::Symmetry;
use sttt::util::board_gen::{random_board_with_forced_win, random_board_with_moves, random_board_with_outcome};

#[test]
fn sttt_empty() {
    test_main(&STTTBoard::default())
}

#[test]
fn sttt_few() {
    test_main(&random_board_with_moves(&STTTBoard::default(), 10, &mut consistent_rng()))
}

#[test]
fn sttt_close() {
    test_main(&random_board_with_forced_win(&STTTBoard::default(), 5, &mut consistent_rng()))
}

#[test]
fn sttt_draw() {
    test_main(&random_board_with_outcome(&STTTBoard::default(), Outcome::Draw, &mut consistent_rng()))
}

#[test]
fn ataxx_empty() {
    test_main(&AtaxxBoard::new_without_gaps())
}

#[test]
fn ataxx_few() {
    test_main(&random_board_with_moves(&AtaxxBoard::new_without_gaps(), 10, &mut consistent_rng()))
}

#[test]
fn ataxx_close() {
    let mut rng = consistent_rng();

    // generate a board that's pretty full instead of the more likely empty board
    let start = random_board_with_moves(&AtaxxBoard::new_without_gaps(), 120, &mut rng);
    let board = random_board_with_forced_win(&start, 5, &mut rng);

    test_main(&board)
}

#[test]
fn ataxx_done() {
    test_main(&random_board_with_outcome(&AtaxxBoard::new_without_gaps(), Outcome::WonBy(Player::A), &mut consistent_rng()))
}

#[test]
fn ataxx_forced_pass() {
    let board = AtaxxBoard::from_fen("xxxxxxx/-------/-------/o6/7/7/7 x");
    assert!(!board.is_done(), "Board is not done, player B can still play");
    test_main(&board)
}

fn test_main<B: Board>(board: &B) {
    if !board.is_done() {
        test_available_match(board);
        test_random_available_uniform(board);
    }

    test_symmetry(board);
}

fn test_available_match<B: Board>(board: &B) {
    println!("available_moves and is_available match:");
    println!("{}", board);

    let all: Vec<B::Move> = B::all_possible_moves().collect();
    let available: Vec<B::Move> = board.available_moves().collect();

    // check that every generated move is indeed available
    for &mv in &available {
        assert!(board.is_available_move(mv), "generated move {:?} is not available", mv);
    }

    // check that every available move is generated
    for &mv in &all {
        if board.is_available_move(mv) {
            assert!(available.contains(&mv), "available move {:?} was not generated", mv);
        }
    }

    // check that there are no duplicates anywhere
    assert_eq!(all.len(), HashSet::<_, RandomState>::from_iter(&all).len(), "Found duplicate move");
    assert_eq!(available.len(), HashSet::<_, RandomState>::from_iter(&available).len(), "Found duplicate move");

    // check that all_possible_moves and available_moves have the same ordering
    let all_filtered = all.iter().copied()
        .filter(|&mv| board.is_available_move(mv))
        .collect_vec();
    assert_eq!(available, all_filtered, "Move order mismatch")
}

/// Test whether the random move distribution is uniform using
/// [Pearson's chi-squared test](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test).
fn test_random_available_uniform<B: Board>(board: &B) {
    assert!(!board.is_done(), "invalid board to test");

    println!("random_available uniform:");
    println!("{}", board);

    let mut rng = consistent_rng();

    let available_move_count = board.available_moves().count();
    let total_samples = 1000 * available_move_count;
    let expected_samples = total_samples as f32 / available_move_count as f32;

    println!("Available moves: {}, samples: {}, expected: {}", available_move_count, total_samples, expected_samples);

    let mut counts: BTreeMap<B::Move, u32> = BTreeMap::new();
    for _ in 0..total_samples {
        let mv = board.random_available_move(&mut rng);
        *counts.entry(mv).or_default() += 1;
    }

    for (&mv, &count) in &counts {
        println!("Move {:?} -> count {} ~ {}", mv, count, count as f32 / expected_samples);
    }

    for (&mv, &count) in &counts {
        assert!((count as f32) > 0.8 * expected_samples, "Move {:?} not generated often enough", mv);
        assert!((count as f32) < 1.2 * expected_samples, "Move {:?} generated too often", mv);
    }
}

fn test_symmetry<B: Board>(board: &B) {
    println!("symmetries:");

    for &sym in B::Symmetry::all() {
        let sym_inv = sym.inverse();

        println!("{:?}", sym);
        println!("inverse: {:?}", sym_inv);

        let mapped = board.map(sym);
        let back = mapped.map(sym_inv);

        // these prints test that the board is consistent enough to print it
        println!("Mapped:\n{}", mapped);
        println!("Back:\n{}", back);

        if sym == B::Symmetry::identity() {
            assert_eq!(board, &mapped);
        }
        assert_eq!(board, &back);

        assert_eq!(board.outcome(), mapped.outcome());
        assert_eq!(board.next_player(), mapped.next_player());

        if !board.is_done() {
            let mut expected_moves: Vec<B::Move> = board.available_moves()
                .map(|c| B::map_move(sym, c))
                .collect();
            let mut actual_moves: Vec<B::Move> = mapped.available_moves()
                .collect();

            expected_moves.sort();
            actual_moves.sort();

            assert_eq!(expected_moves, actual_moves);

            for mv in actual_moves {
                assert!(mapped.is_available_move(mv));
            }
        }
    }
}

fn consistent_rng() -> impl Rng {
    Xoroshiro64StarStar::seed_from_u64(0)
}
