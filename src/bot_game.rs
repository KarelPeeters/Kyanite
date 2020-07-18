use rand::{Rng, thread_rng};

use crate::board::{Board, Coord, Player};
use crate::mcts::old_move_mcts;

pub trait Bot {
    fn play(&self, board: &Board) -> Option<Coord>;
}

impl<F: Fn(&Board) -> Option<Coord>> Bot for F {
    fn play(&self, board: &Board) -> Option<Coord> {
        self(board)
    }
}

pub struct MCTSBot {
    iterations: usize,
}

impl MCTSBot {
    pub fn new(iterations: usize) -> Self {
        MCTSBot { iterations }
    }
}

impl Bot for MCTSBot {
    fn play(&self, board: &Board) -> Option<Coord> {
        old_move_mcts(board, self.iterations, &mut thread_rng(), false)
    }
}

pub struct RandomBot;

impl Bot for RandomBot {
    fn play(&self, board: &Board) -> Option<Coord> {
        board.random_available_move(&mut thread_rng())
    }
}

pub fn run<A: Bot, B: Bot, R: Rng>(
    bot_a: &A,
    bot_b: &B,
    games: usize,
    shuffle: bool,
    rand: &mut R,
) -> BotGameResult {
    let bots: Vec<&Bot> = vec![bot_a, bot_b];
    let mut wins = [0, 0];

    for i in 0..games {
        println!("Starting game {}/{}", i, games);

        let flip = if shuffle { rand.gen::<bool>() as usize } else { 0 };
        let mut board = Board::new();

        for i in 0.. {
            if board.is_done() {
                break;
            }

            let bot = bots[flip ^ (i % 2)];
            let mv = bot.play(&board).expect("bot didn't return move in unfinished game");
            board.play(mv);
        }

        match board.won_by.unwrap() {
            Player::Player => wins[flip] += 1,
            Player::Enemy => wins[1 - flip] += 1,
            Player::Neutral => {}
        }
    }

    BotGameResult::new(wins[0], wins[1], games)
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