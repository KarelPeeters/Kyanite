use std::str::FromStr;

use board_game::board::{Board, BoardMoves};
use board_game::games::dummy::DummyGame;
use board_game::games::sttt::STTTBoard;
use internal_iterator::InternalIterator;

use kz_core::network::dummy::DummyNetwork;
use kz_core::oracle::DummyOracle;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::FpuMode;
use kz_core::zero::tree::Tree;
use kz_core::zero::wrapper::ZeroSettings;

#[test]
fn tree_smaller_than_batch() {
    let board = DummyGame::from_str("(AA(AB))").unwrap();

    let settings = ZeroSettings::new(16, UctWeights::default(), false, FpuMode::Parent);

    // we're testing that this does not get stuck once the tree runs out
    // it should also stop early enough without visiting the rest of the tree a crazy amount of times,
    //   but that's hard to test)
    let _ = settings.build_tree(&board, &mut DummyNetwork, &DummyOracle, |tree| tree.root_visits() > 100);
}

#[test]
fn empty_tree_display() {
    let board = STTTBoard::default();
    let tree = Tree::new(board);

    println!("{}", tree.display(100, false, 100, true));
}

#[test]
fn keep_tree() {
    let board = STTTBoard::default();
    let settings = ZeroSettings::new(1, UctWeights::default(), false, FpuMode::Parent);
    let visits = 100;

    let parent_tree = settings.build_tree(&board, &mut DummyNetwork, &DummyOracle, |tree| {
        tree.root_visits() >= visits
    });

    println!("{}", parent_tree.display(100, true, 200, true));

    let mv = board.available_moves().next().unwrap();
    let actual_child_tree = parent_tree.keep_moves(&[mv]).unwrap();

    let actual_string = actual_child_tree.display(100, true, 200, true).to_string();
    println!("{}", actual_string);

    let expected_child_tree = settings.build_tree(&board.clone_and_play(mv), &mut DummyNetwork, &mut DummyOracle, |tree| {
        tree.root_visits() >= actual_child_tree.root_visits()
    });

    let expected_string = expected_child_tree.display(100, true, 200, true).to_string();
    println!("{}", expected_string);

    assert_eq!(actual_string, expected_string);
}