use board_game::games::ttt::TTTBoard;
use rand::thread_rng;

use alpha_zero::network::dummy::DummyNetwork;
use alpha_zero::zero::{zero_build_tree, ZeroSettings};

fn main() {
    let board = TTTBoard::default();
    let settings = ZeroSettings::new(1, 2.0, false);
    let tree = zero_build_tree(&board, 100_000_000, settings, &mut DummyNetwork, &mut thread_rng(), || false);

    println!("{}", tree.display(1))
}