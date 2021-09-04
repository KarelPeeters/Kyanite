use std::fmt::Debug;
use std::fmt::Write;
use std::ops::Add;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use itertools::Itertools;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use crate::ai::Bot;
use crate::board::{Board, Outcome};

#[must_use]
pub fn run<B: Board, L: Bot<B>, R: Bot<B>>(
    start: impl Fn() -> B + Sync,
    bot_l: impl Fn() -> L + Sync,
    bot_r: impl Fn() -> R + Sync,
    games_per_side: u32,
    both_sides: bool,
    print_progress_every: Option<u32>,
) -> BotGameResult {
    // this instantiates both both at least once so we catch errors before starting a bunch of threads
    let debug_l = debug_to_sting(&bot_l());
    let debug_r = debug_to_sting(&bot_r());

    let progress_counter = AtomicU32::default();
    let game_count = if both_sides { 2 * games_per_side } else { games_per_side };

    let starts = (0..games_per_side).map(|_| start()).collect_vec();

    let result: ReductionResult = (0..games_per_side).into_par_iter().panic_fuse().map(|game_i| {
        let pair_i = if both_sides { game_i / 2 } else { game_i };
        let start = starts[pair_i as usize].clone();

        let mut bot_l = bot_l();
        let mut bot_r = bot_r();

        let mut total_time_l = 0.0;
        let mut total_time_r = 0.0;
        let mut move_count_l: u32 = 0;
        let mut move_count_r: u32 = 0;

        let flip = if both_sides { game_i % 2 == 1 } else { false };
        let mut board = start;
        let player_first = board.next_player();

        for move_i in 0.. {
            if board.is_done() {
                break;
            }

            let start = Instant::now();
            let mv = if flip ^ (move_i % 2 == 0) {
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
                println!("Progress: {}", progress as f32 / games_per_side as f32);
            }
        }

        let outcome = board.outcome().unwrap();
        let win_first = (outcome == Outcome::WonBy(player_first)) as u32;
        let win_second = (outcome == Outcome::WonBy(player_first.other())) as u32;

        let (wins_l, wins_r) = if flip { (win_second, win_first) } else { (win_first, win_second) };

        ReductionResult { wins_l, wins_r, total_time_l, total_time_r, move_count_l, move_count_r }
    }).reduce(ReductionResult::default, ReductionResult::add);

    let draws = game_count - result.wins_l - result.wins_r;
    let score_l = (result.wins_l as f32 + 0.5 * draws as f32) / (game_count as f32);
    let elo = -400.0 * (1.0 / score_l - 1.0).log10();

    BotGameResult {
        game_count,
        game_length: (result.move_count_l + result.move_count_r) as f32 / (game_count) as f32,
        win_rate_l: (result.wins_l as f32) / (game_count as f32),
        draw_rate: (draws as f32) / (game_count as f32),
        win_rate_r: (result.wins_r as f32) / (game_count as f32),
        elo_l: elo,
        time_l: result.total_time_l / (result.move_count_l as f32),
        time_r: result.total_time_r / (result.move_count_r as f32),
        debug_l,
        debug_r,
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
pub struct BotGameResult {
    pub game_count: u32,

    //average game length, the total number of moves played per game
    pub game_length: f32,

    //wdl
    pub win_rate_l: f32,
    pub draw_rate: f32,
    pub win_rate_r: f32,

    //elo of the left player, assuming the right elo is 0
    pub elo_l: f32,

    //time per move in seconds
    pub time_l: f32,
    pub time_r: f32,

    // bot debug strings
    pub debug_l: String,
    pub debug_r: String,
}

fn debug_to_sting(d: &impl Debug) -> String {
    let mut s = String::new();
    write!(&mut s, "{:?}", d).unwrap();
    s
}