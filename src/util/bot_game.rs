use std::ops::Add;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use crate::ai::Bot;
use crate::board::{Board, Outcome, Player};

pub fn run<B: Board, L: Bot<B>, R: Bot<B>>(
    start: impl Fn() -> B + Sync,
    bot_l: impl Fn() -> L + Sync,
    bot_r: impl Fn() -> R + Sync,
    games: u32,
    shuffle: bool,
    print_progress_every: Option<u32>,
) -> BotGameResult {
    let progress_counter = AtomicU32::default();

    let result: ReductionResult = (0..games).into_par_iter().map(|_i| {
        let mut bot_l = bot_l();
        let mut bot_r = bot_r();

        let mut total_time_l = 0.0;
        let mut total_time_r = 0.0;
        let mut move_count_l: u32 = 0;
        let mut move_count_r: u32 = 0;

        let mut rand = SmallRng::from_entropy();

        let flip = if shuffle { rand.gen::<bool>() } else { false };
        let mut board = start();

        for i in 0.. {
            if board.is_done() {
                break;
            }

            let start = Instant::now();
            let mv = if flip ^ (i % 2 == 0) {
                let mv = bot_l.select_move(&board);
                total_time_l += (Instant::now() - start).as_secs_f32();
                move_count_l += 1;
                mv
            } else {
                let mv = bot_r.select_move(&board);
                total_time_r += (Instant::now() - start).as_secs_f32();
                move_count_r += 1;
                mv
            };

            board.play(mv);
        }

        if let Some(print_progress) = print_progress_every {
            let progress = progress_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if progress % print_progress == 0 {
                println!("Progress: {}", progress as f32 / games as f32);
            }
        }

        let outcome = board.outcome().unwrap();
        let win_a = (outcome == Outcome::WonBy(Player::A)) as u32;
        let win_b = (outcome == Outcome::WonBy(Player::B)) as u32;

        let (wins_l, wins_r) = if flip { (win_b, win_a) } else { (win_a, win_b) };

        ReductionResult { wins_l, wins_r, total_time_l, total_time_r, move_count_l, move_count_r }
    }).reduce(ReductionResult::default, ReductionResult::add);

    let draws = games - result.wins_l - result.wins_r;
    BotGameResult {
        games,
        wins_l: result.wins_l,
        wins_r: result.wins_r,
        draws,
        win_rate_l: (result.wins_l as f32) / (games as f32),
        win_rate_r: (result.wins_r as f32) / (games as f32),
        draw_rate: (draws as f32) / (games as f32),
        time_l: result.total_time_l / (result.move_count_l as f32),
        time_r: result.total_time_r / (result.move_count_r as f32),
    }
}

#[derive(Default, Debug, Copy, Clone)]
struct ReductionResult {
    wins_l: u32,
    wins_r: u32,
    total_time_l: f32,
    total_time_r: f32,
    move_count_l: u32,
    move_count_r: u32,
}

impl std::ops::Add for ReductionResult {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        ReductionResult {
            wins_l: self.wins_l + rhs.wins_l,
            wins_r: self.wins_r + rhs.wins_r,
            total_time_l: self.total_time_l + rhs.total_time_l,
            total_time_r: self.total_time_r + rhs.total_time_r,
            move_count_l: self.move_count_l + rhs.move_count_l,
            move_count_r: self.move_count_r + rhs.move_count_r,
        }
    }
}

#[derive(Debug)]
#[must_use]
pub struct BotGameResult {
    games: u32,
    wins_l: u32,
    wins_r: u32,
    draws: u32,

    win_rate_l: f32,
    win_rate_r: f32,
    draw_rate: f32,

    //time per move in seconds
    time_l: f32,
    time_r: f32,
}
