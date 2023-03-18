use std::str::FromStr;

use board_game::board::{Board, BoardMoves};
use board_game::games::dummy::DummyGame;
use board_game::games::sttt::STTTBoard;
use internal_iterator::InternalIterator;
use rand::rngs::StdRng;
use rand::SeedableRng;

use kz_core::network::dummy::DummyNetwork;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::{FpuMode, QMode};
use kz_core::zero::tree::Tree;
use kz_core::zero::wrapper::ZeroSettings;

#[test]
fn tree_smaller_than_batch() {
    let board = DummyGame::from_str("(AA(AB))").unwrap();

    let settings = ZeroSettings::simple(16, UctWeights::default(), QMode::wdl(), FpuMode::Relative(0.0));

    // we're testing that this does not get stuck once the tree runs out
    // it should also stop early enough without visiting the rest of the tree a crazy amount of times,
    //   but that's hard to test
    let mut rng = StdRng::seed_from_u64(0);
    let tree = settings.build_tree(&board, &mut DummyNetwork, &mut rng, |tree| tree.root_visits() > 100);
    println!("{}", tree.display(1, true, usize::MAX, true));
}

#[test]
fn empty_tree_display() {
    let board = STTTBoard::default();
    let tree = Tree::new(board);

    println!("{}", tree.display(100, false, 100, true));
}

//TODO fix this test again, maybe best_move is influenced by randomness?
#[ignore]
#[test]
fn keep_tree() {
    let board = STTTBoard::default();
    let settings = ZeroSettings::simple(1, UctWeights::default(), QMode::wdl(), FpuMode::Relative(0.0));
    let visits = 100;
    let mut rng = StdRng::seed_from_u64(0);

    let parent_tree = settings.build_tree(&board, &mut DummyNetwork, &mut rng, |tree| tree.root_visits() >= visits);

    println!("{}", parent_tree.display(100, true, 200, true));

    let mv = board.available_moves().unwrap().next().unwrap();
    let actual_child_tree = parent_tree.keep_moves(&[mv]).unwrap();

    let actual_string = actual_child_tree.display(100, true, 200, true).to_string();
    println!("{}", actual_string);

    let expected_child_tree = settings.build_tree(&board.clone_and_play(mv).unwrap(), &mut DummyNetwork, &mut rng, |tree| {
        tree.root_visits() >= actual_child_tree.root_visits()
    });

    let expected_string = expected_child_tree.display(100, true, 200, true).to_string();
    println!("{}", expected_string);

    assert_eq!(actual_string, expected_string);
}
