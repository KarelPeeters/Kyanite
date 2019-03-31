use std::f64;

use crate::board::{Board, Coord, Player};

pub fn move_minimax(board: &Board, depth: u32) -> Option<Coord> {
    negamax(board, value(board), depth,
            f64::NEG_INFINITY, f64::INFINITY,
            player_sign(board.next_player),
    ).0
}

pub fn negamax(board: &Board, c_value: f64, depth: u32, a: f64, b: f64, player: f64) -> (Option<Coord>, f64) {
    if depth == 0 || board.is_done() {
        return (board.last_move, player * c_value);
    }

    let mut best_value = f64::NEG_INFINITY;
    let mut best_move: Option<Coord> = None;
    let mut new_a = a;

    for mv in board.available_moves() {
        let mut child = board.clone();

        //Calculate the new score
        let mut child_value = c_value + TILE_VALUE * factor(mv.om()) * factor(mv.os()) * player;
        if child.play(mv) {
            if child.is_done() {
                child_value = f64::INFINITY * player;
            } else {
                child_value += MACRO_VALUE * factor(mv.om()) * player;
            }
        }

        //Check if the (global) value of this child is better then the previous best child
        let value = -negamax(&child, child_value, depth - 1, -b, -new_a, -player).1;
        if value > best_value || best_move.is_none() {
            best_value = value;
            best_move = Some(mv);
        }
        new_a = new_a.max(value);
        if new_a >= b {
            break;
        }
    }

    (best_move, best_value)
}

pub fn value(board: &Board) -> f64 {
    match board.won_by {
        Some(player) => f64::INFINITY * player_sign(player),
        _ => {
            let tiles: f64 = (0..81).map(|c| factor(c % 9) * factor(c / 9) * player_sign(board.tile(Coord::of_o(c)))).sum();
            let macros: f64 = (0..9).map(|c| factor(c) * player_sign(board.macr(c))).sum();
            TILE_VALUE * tiles + MACRO_VALUE * macros
        }
    }
}

fn player_sign(player: Player) -> f64 {
    match player {
        Player::Player => 1.0,
        Player::Enemy => -1.0,
        Player::Neutral => 0.0,
    }
}

fn factor(os: u8) -> f64 {
    match os {
        4 => CENTER_FACTOR,
        _ if os % 2 == 0 => CORNER_FACTOR,
        _ => EDGE_FACTOR,
    }
}

const TILE_VALUE: f64 = 1.0;
const MACRO_VALUE: f64 = 10e9;

const CENTER_FACTOR: f64 = 4.0;
const CORNER_FACTOR: f64 = 3.0;
const EDGE_FACTOR: f64 = 1.0;