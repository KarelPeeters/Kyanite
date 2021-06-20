use std::ops::Add;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use crate::board::{Board, Coord, Player};
use std::sync::atomic::{AtomicU32, Ordering};

pub trait Bot {
    /// Decide which move to play in the given board.
    /// If there are no moves (because `board.is_done()`) return `None`.
    fn play(&mut self, board: &Board) -> Option<Coord>;
}

impl<F: FnMut(&Board) -> Option<Coord>> Bot for F {
    fn play(&mut self, board: &Board) -> Option<Coord> {
        self(board)
    }
}

pub fn run<A: Bot, B: Bot>(
    bot_a: impl Fn() -> A + Sync,
    bot_b: impl Fn() -> B + Sync,
    games: u32,
    shuffle: bool,
    print_progress: Option<u32>,
) -> BotGameResult {
    let progress_counter = AtomicU32::default();

    let result: ReductionResult = (0..games).into_par_iter().map(|_i| {
        let mut bot_a = bot_a();
        let mut bot_b = bot_b();

        let mut total_time_a = 0.0;
        let mut total_time_b = 0.0;
        let mut move_count_a: u32 = 0;
        let mut move_count_b: u32 = 0;

        let mut rand = SmallRng::from_entropy();

        let flip = if shuffle { rand.gen::<bool>() } else { false };
        let mut board = Board::new();

        for i in 0.. {
            if board.is_done() {
                break;
            }

            let start = Instant::now();
            let mv = if flip ^ (i % 2 == 0) {
                let mv = bot_a.play(&board).expect("bot A didn't return move in unfinished game");
                total_time_a += (Instant::now() - start).as_secs_f32();
                move_count_a += 1;
                mv
            } else {
                let mv = bot_b.play(&board).expect("bot B didn't return move in unfinished game");
                total_time_b += (Instant::now() - start).as_secs_f32();
                move_count_b += 1;
                mv
            };

            board.play(mv);
        }

        let (win_x, win_o) = match board.won_by.unwrap() {
            Player::X => (1, 0),
            Player::O => (0, 1),
            Player::Neutral => (0, 0)
        };

        let (win_a, win_b) = if flip { (win_o, win_x) } else { (win_x, win_o) };

        if let Some(print_progress) = print_progress {
            let progress = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if progress % print_progress == 0 {
                println!("Progress: {}", progress as f32 / games as f32);
            }
        }

        ReductionResult { wins_a: win_a, wins_b: win_b, total_time_a, total_time_b, move_count_a, move_count_b }
    }).reduce(ReductionResult::default, ReductionResult::add);

    let ties = games - result.wins_a - result.wins_b;
    BotGameResult {
        games,
        wins_a: result.wins_a,
        wins_b: result.wins_b,
        ties,
        rate_a: (result.wins_a as f32) / (games as f32),
        rate_b: (result.wins_b as f32) / (games as f32),
        rate_tie: (ties as f32) / (games as f32),
        time_a: result.total_time_a / (result.move_count_a as f32),
        time_b: result.total_time_b / (result.move_count_b as f32),
    }
}

#[derive(Default, Debug, Copy, Clone)]
struct ReductionResult {
    wins_a: u32,
    wins_b: u32,
    total_time_a: f32,
    total_time_b: f32,
    move_count_a: u32,
    move_count_b: u32,
}

impl std::ops::Add for ReductionResult {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        ReductionResult {
            wins_a: self.wins_a + rhs.wins_a,
            wins_b: self.wins_b + rhs.wins_b,
            total_time_a: self.total_time_a + rhs.total_time_a,
            total_time_b: self.total_time_b + rhs.total_time_b,
            move_count_a: self.move_count_a + rhs.move_count_a,
            move_count_b: self.move_count_b + rhs.move_count_b,
        }
    }
}

#[derive(Debug)]
#[must_use]
pub struct BotGameResult {
    games: u32,
    wins_a: u32,
    wins_b: u32,
    ties: u32,

    rate_a: f32,
    rate_b: f32,
    rate_tie: f32,

    //time per move in seconds
    time_a: f32,
    time_b: f32,
}
