use rand::Rng;

use crate::board::{Board, Coord, Player};

pub trait Bot {
    fn play(&self, board: &Board) -> Option<Coord>;
}

impl<F: Fn(&Board) -> Option<Coord>> Bot for F {
    fn play(&self, board: &Board) -> Option<Coord> {
        self(board)
    }
}

pub fn run<A: Bot, B: Bot, R: Rng>(bot_a: &A, bot_b: &B, games: usize, shuffle: bool, rand: &mut R)
                                   -> BotGameResult {
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
pub struct BotGameResult {
    wins_a: usize,
    wins_b: usize,
    ties: usize,
    games: usize,

    rate_a: f32,
    rate_b: f32,
    rate_tie: f32,
}

impl BotGameResult {
    fn new(wins_a: usize, wins_b: usize, games: usize) -> BotGameResult {
        let ties = games - wins_b - wins_b;
        BotGameResult {
            wins_a,
            wins_b,
            ties,
            games,
            rate_a: wins_a as f32 / games as f32,
            rate_b: wins_b as f32 / games as f32,
            rate_tie: ties as f32 / games as f32,
        }
    }
}