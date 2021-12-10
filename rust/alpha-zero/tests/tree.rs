use std::str::FromStr;

use board_game::games::dummy::DummyGame;

use alpha_zero::network::dummy::DummyNetwork;
use alpha_zero::oracle::DummyOracle;
use alpha_zero::zero::step::FpuMode;
use alpha_zero::zero::wrapper::ZeroSettings;

#[test]
fn tree_smaller_than_batch() {
    let game = DummyGame::from_str("(AA(AB))").unwrap();

    let settings = ZeroSettings::new(16, 2.0, false, FpuMode::Parent);

    // we're testing that this does not get stuck once the tree runs out
    // it should also stop early enough without visiting the rest of the tree a crazy amount of times,
    //   but that's hard to test)
    let _ = settings.build_tree(&game, &mut DummyNetwork, &DummyOracle, |tree| tree.root_visits() > 100);
}