use std::fmt::Debug;

use board_game::board::Board;

use crate::zero::tree::Tree;

pub trait StopCondition<B: Board> {
    type D: Debug + ?Sized;

    fn should_stop(&self, tree: &Tree<B>) -> bool;

    fn debug(&self) -> &Self::D;
}

impl<B: Board, F: Fn(&Tree<B>) -> bool> StopCondition<B> for F {
    type D = str;

    fn should_stop(&self, tree: &Tree<B>) -> bool {
        self(tree)
    }

    fn debug(&self) -> &str {
        "custom"
    }
}

impl<B: Board> StopCondition<B> for u64 {
    type D = u64;

    fn should_stop(&self, tree: &Tree<B>) -> bool {
        tree.root_visits() >= *self
    }

    fn debug(&self) -> &u64 {
        self
    }
}
