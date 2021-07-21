use std::fmt::{Debug, Formatter};

use internal_iterator::InternalIterator;
use rand::Rng;

use crate::ai::Bot;
use crate::board::Board;
use crate::wdl::POV;

pub struct RandomBot<R: Rng> {
    rng: R,
}

impl<R: Rng> Debug for RandomBot<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RandomBot")
    }
}

impl<R: Rng> RandomBot<R> {
    pub fn new(rng: R) -> Self {
        RandomBot { rng }
    }
}

impl<B: Board, R: Rng> Bot<B> for RandomBot<R> {
    fn select_move(&mut self, board: &B) -> B::Move {
        board.random_available_move(&mut self.rng)
    }
}

pub struct RolloutBot<R: Rng> {
    rollouts: u32,
    rng: R,
}

impl<R: Rng> Debug for RolloutBot<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RolloutBot {{ rollouts: {} }}", self.rollouts)
    }
}

impl<R: Rng> RolloutBot<R> {
    pub fn new(rollouts: u32, rng: R) -> Self {
        RolloutBot { rollouts, rng }
    }
}

impl<B: Board, R: Rng> Bot<B> for RolloutBot<R> {
    fn select_move(&mut self, board: &B) -> B::Move {
        let rollouts_per_move = self.rollouts / board.available_moves().count() as u32;

        board.available_moves().max_by_key(|&mv| {
            let child = board.clone_and_play(mv);

            let score: i64 = (0..rollouts_per_move).map(|_| {
                let mut copy = child.clone();
                while !copy.is_done() {
                    copy.play(copy.random_available_move(&mut self.rng))
                }
                copy.outcome().unwrap().pov(board.next_player()).sign::<i64>()
            }).sum();

            score
        }).unwrap()
    }
}