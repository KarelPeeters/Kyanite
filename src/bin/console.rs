use std::io;
use std::io::Write;

use rand::rngs::SmallRng;
use rand::SeedableRng;
use regex::Regex;

use sttt::board::{Board, Coord, Player};
use sttt::bot_game::Bot;
use sttt::mcts::MCTSBot;

fn main() {
    console_game(MCTSBot::new(10_000, 2.0, SmallRng::from_entropy()))
}

fn console_game<B: Bot>(mut bot: B) {
    let move_regex = Regex::new(r"^(?P<om>\d+)\s*(?:,\s*)?(?P<os>\d+)$").unwrap();

    let mut history = Vec::new();
    let mut board = Board::new();

    println!("{}", board);

    let mut line = String::new();
    let user_player = Player::X;

    loop {
        //Player move
        'playerMove: loop {
            print!("Play move: ");
            io::stdout().flush().expect("Could not flush stdout");

            line.clear();
            io::stdin().read_line(&mut line).unwrap();
            let line = line.trim();

            if line == "u" {
                board = match history.pop() {
                    Some(board) => {
                        println!("Undo");
                        println!("{}", board);
                        board
                    }
                    None => {
                        println!("No history");
                        board
                    }
                }
            } else if let Some(m) = move_regex.captures(&line) {
                let om: u8 = m["om"].parse().unwrap();
                let os: u8 = m["os"].parse().unwrap();

                if om <= 8 && os <= 8 {
                    let mv = Coord::from_oo(om, os);
                    if board.is_available_move(mv) {
                        history.push(board.clone());
                        board.play(mv);
                        println!("{}", board);
                        break 'playerMove;
                    } else {
                        eprintln!("Move not available")
                    }
                } else {
                    eprintln!("Illegal value")
                }
            } else {
                eprintln!("Invalid move format")
            }
        }

        if board.is_done() {
            break;
        }

        //Bot move
        let mv = bot.play(&board)
            .expect("Bot should return move");

        board.play(mv);
        println!("{}", board);

        if board.is_done() {
            break;
        }
    }

    match board.won_by {
        None => panic!("Game should be finished"),
        Some(Player::Neutral) => println!("You drew :|"),
        Some(player) => {
            if player == user_player {
                println!("You won :)")
            } else {
                println!("You lost :(")
            }
        }
    }
}