use rand::{Rng, SeedableRng, thread_rng};
use rand::rngs::SmallRng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use crate::board::{Board, Coord, Player};
use std::sync::atomic::{AtomicUsize, Ordering};

pub trait Bot {
    fn play(&mut self, board: &Board) -> Option<Coord>;
}

impl<F: FnMut(&Board) -> Option<Coord>> Bot for F {
    fn play(&mut self, board: &Board) -> Option<Coord> {
        self(board)
    }
}

pub struct RandomBot;

impl Bot for RandomBot {
    fn play(&mut self, board: &Board) -> Option<Coord> {
        board.random_available_move(&mut thread_rng())
    }
}

pub fn run<A: Bot, B: Bot>(
    bot_a: impl Fn() -> A + Sync,
    bot_b: impl Fn() -> B + Sync,
    games: usize,
    shuffle: bool,
) -> BotGameResult {
    let progress_counter = AtomicUsize::default();

    let score = (0..games).into_par_iter().map(|_i| {
        let mut bot_a = bot_a();
        let mut bot_b = bot_b();

        let mut rand = SmallRng::from_entropy();

        let flip = if shuffle { rand.gen::<bool>() } else { false };
        let mut board = Board::new();

        for i in 0.. {
            if board.is_done() {
                break;
            }

            let mv = if flip ^ (i % 2 == 0) {
                bot_a.play(&board).expect("bot A didn't return move in unfinished game")
            } else {
                bot_b.play(&board).expect("bot B didn't return move in unfinished game")
            };

            board.play(mv);
        }

        let score = match board.won_by.unwrap() {
            Player::X => (1, 0),
            Player::O => (0, 1),
            Player::Neutral => (0, 0)
        };

        let score = if flip { (score.1, score.0) } else { score };

        let progress = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
        if progress % (games / 10) == 0 {
            println!("Progress: {}", progress as f32 / games as f32);
        }

        score
    }).reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

    BotGameResult::new(score.0, score.1, games)
}

#[allow(dead_code)]
#[derive(Debug)]
#[must_use]
pub struct BotGameResult {
    games: usize,
    wins_a: usize,
    wins_b: usize,
    ties: usize,

    rate_a: f32,
    rate_b: f32,
    rate_tie: f32,
}

impl BotGameResult {
    fn new(wins_a: usize, wins_b: usize, games: usize) -> BotGameResult {
        let ties = games - wins_a - wins_b;
        BotGameResult {
            games,
            wins_a,
            wins_b,
            ties,
            rate_a: wins_a as f32 / games as f32,
            rate_b: wins_b as f32 / games as f32,
            rate_tie: ties as f32 / games as f32,
        }
    }
}