use decorum::NotNan;
use rand::Rng;

use crate::ai::Bot;
use crate::board::Board;

pub struct RandomBot<R: Rng> {
    rng: R,
}

impl<B: Board, R: Rng> Bot<B> for RandomBot<R> {
    fn select_move(&mut self, board: &B) -> B::Move {
        board.random_available_move(&mut self.rng)
    }
}

pub struct RolloutBot<R: Rng> {
    rng: R,
    rollouts_per_move: u32,
}

impl<B: Board, R: Rng> Bot<B> for RolloutBot<R> {
    fn select_move(&mut self, board: &B) -> B::Move {
        board.available_moves().max_by_key(|&mv| {
            let child = board.clone_and_play(mv);

            let score: f32 = (0..self.rollouts_per_move).map(|_| {
                let mut copy = child.clone();
                while !copy.is_done() {
                    copy.play(copy.random_available_move(&mut self.rng))
                }
                copy.outcome().unwrap().sign(board.next_player())
            }).sum();

            NotNan::from(score)
        }).unwrap()
    }
}