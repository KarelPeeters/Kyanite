use crate::bot_game::Bot;
use crate::board::{Board, Coord};
use rand::{thread_rng, Rng};

pub struct RandomBot<R: Rng>(R);

impl<R: Rng> RandomBot<R> {
    pub fn new(rng: R) -> Self {
        RandomBot(rng)
    }
}

impl<R: Rng> Bot for RandomBot<R> {
    fn play(&mut self, board: &Board) -> Option<Coord> {
        board.random_available_move(&mut self.0)
    }
}

pub struct RolloutBot {
    rollouts: usize
}

impl RolloutBot {
    pub fn new(rollouts: usize) -> Self {
        RolloutBot { rollouts }
    }
}

impl Bot for RolloutBot {
    fn play(&mut self, board: &Board) -> Option<Coord> {
        let mut rand = thread_rng();
        let rollouts = self.rollouts / board.available_moves().count();

        board.available_moves().max_by_key(|&mv| {
            let mut played_copy = board.clone();
            played_copy.play(mv);

            let score: i32 = (0..rollouts).map(|_| {
                let mut rollout_copy = played_copy.clone();

                while let Some(mv) = rollout_copy.random_available_move(&mut rand) {
                    rollout_copy.play(mv);
                }

                match rollout_copy.won_by.unwrap() {
                    player if player == board.next_player => 1,
                    player if player == board.next_player.other() => -1,
                    _ => 0,
                }
            }).sum();

            score
        })
    }
}