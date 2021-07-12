use std::collections::HashMap;

use internal_iterator::InternalIterator;
use mathru::statistics::distrib::{ChiSquare, Continuous};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoroshiro64StarStar;

use sttt::board::{Board, Outcome, Player};
use sttt::games::ataxx::AtaxxBoard;
use sttt::games::sttt::STTTBoard;
use sttt::symmetry::Symmetry;
use sttt::util::board_gen::{random_board_with_forced_win, random_board_with_moves, random_board_with_outcome};

#[test]
fn sttt_empty() {
    println!("derp");
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
    test_main(&AtaxxBoard::new_without_blocks())
}

#[test]
fn ataxx_few() {
    test_main(&random_board_with_moves(&AtaxxBoard::new_without_blocks(), 10, &mut consistent_rng()))
}

#[test]
fn ataxx_close() {
    let mut rng = consistent_rng();

    // generate a board that's pretty full instead of the more likely empty board
    let start = random_board_with_moves(&AtaxxBoard::new_without_blocks(), 120, &mut rng);
    let board = random_board_with_forced_win(&start, 5, &mut rng);

    test_main(&board)
}

#[test]
fn ataxx_done() {
    test_main(&random_board_with_outcome(&AtaxxBoard::new_without_blocks(), Outcome::WonBy(Player::A), &mut consistent_rng()))
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

    let available: Vec<B::Move> = board.available_moves().collect();

    // check that every generated move is indeed available
    for &mv in &available {
        assert!(board.is_available_move(mv));
    }

    //check that every available move is generated
    B::all_possible_moves().for_each(|mv: B::Move| {
        if board.is_available_move(mv) {
            assert!(available.contains(&mv));
        }
    })
}

/// Test whether the random move distribution is uniform using
/// [Pearson's chi-squared test](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test).
fn test_random_available_uniform<B: Board>(board: &B) {
    println!("random_available uniform:");
    println!("{}", board);

    let mut rng = consistent_rng();

    let total_samples = 50 * B::all_possible_moves().count();
    println!("Sampling {} random moves", total_samples);

    let move_count = board.available_moves().count();
    let samples_per_move = total_samples as f32 / move_count as f32;

    let mut counts: HashMap<B::Move, u64> = HashMap::new();
    for _ in 0..total_samples {
        let mv = board.random_available_move(&mut rng);
        *counts.entry(mv).or_default() += 1;
    }

    board.available_moves().for_each(|mv| {
        // every move needs to be generated at least once
        assert!(counts.contains_key(&mv));
    });

    println!("Counts: {}", counts.len());
    let mut i = 0;

    board.available_moves().for_each(|mv: B::Move| {
        let count = counts.get(&mv).unwrap();
        println!("  move {:?} -> {}", mv, count);
    });

    let x2: f32 = counts.iter()
        .map(|(k, &c)| {
            i += 1;
            println!("{} {:?} {}", i, k, c);
            (c as f32 - samples_per_move).powi(2) / samples_per_move
        })
        .sum::<f32>();
    let p = 1.0 - ChiSquare::new(1).cdf(x2);
    println!("x2={} -> p={}", x2, p);

    //TODO figure out why this is not working yet
    // assert!(p < 0.01, "Distribution is not uniform enough");
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
