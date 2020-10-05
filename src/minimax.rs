use derive_more::Constructor;

use crate::board::{Board, Coord, Player};
use crate::bot_game::Bot;

#[derive(Constructor)]
pub struct MiniMaxBot {
    depth: u32
}

impl Bot for MiniMaxBot {
    fn play(&mut self, board: &Board) -> Option<Coord> {
        evaluate_minimax(board, self.depth).best_move
    }
}

pub struct Evaluation {
    pub best_move: Option<Coord>,
    pub value: i64,
}

fn evaluate_minimax(board: &Board, depth: u32) -> Evaluation {
    negamax(board, value(board), depth,
            -BOUND_VALUE, BOUND_VALUE,
            player_sign(board.next_player),
    )
}

fn negamax(board: &Board, c_value: i64, depth: u32, a: i64, b: i64, player: i64) -> Evaluation {
    if depth == 0 || board.is_done() {
        return Evaluation {
            best_move: board.last_move,
            value: player * c_value,
        };
    }

    let mut best_value = -BOUND_VALUE;
    let mut best_move: Option<Coord> = None;
    let mut new_a = a;

    for mv in board.available_moves() {
        let mut child = board.clone();

        //Calculate the new score
        let mut child_value = c_value + TILE_VALUE * factor(mv.om()) * factor(mv.os()) * player;
        if child.play(mv) {
            if child.is_done() {
                child_value = (WIN_VALUE + depth as i64) * player;
            } else {
                child_value += MACRO_VALUE * factor(mv.om()) * player;
            }
        }

        //Check if the (global) value of this child is better then the previous best child
        let value = -negamax(&child, child_value, depth - 1, -b, -new_a, -player).value;
        if value > best_value || best_move.is_none() {
            best_value = value;
            best_move = Some(mv);
        }
        new_a = new_a.max(value);
        if new_a >= b {
            break;
        }
    }

    Evaluation {
        best_move,
        value: best_value,
    }
}

pub fn value(board: &Board) -> i64 {
    match board.won_by {
        Some(player) => WIN_VALUE * player_sign(player),
        _ => {
            let tiles: i64 = (0..81).map(|c| factor(c % 9) * factor(c / 9) * player_sign(board.tile(Coord::from_o(c)))).sum();
            let macros: i64 = (0..9).map(|c| factor(c) * player_sign(board.macr(c))).sum();
            TILE_VALUE * tiles + MACRO_VALUE * macros
        }
    }
}

fn player_sign(player: Player) -> i64 {
    match player {
        Player::X => 1,
        Player::O => -1,
        Player::Neutral => 0,
    }
}

fn factor(os: u8) -> i64 {
    match os {
        4 => CENTER_FACTOR,
        _ if os % 2 == 0 => CORNER_FACTOR,
        _ => EDGE_FACTOR,
    }
}

const TILE_VALUE: i64 = 1;
const MACRO_VALUE: i64 = 1_000;
const WIN_VALUE: i64 = 1_000_000;
const BOUND_VALUE: i64 = 1_000_000_000;

const CENTER_FACTOR: i64 = 4;
const CORNER_FACTOR: i64 = 3;
const EDGE_FACTOR: i64 = 1;