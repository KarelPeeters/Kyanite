use std::io::{BufRead, stdin};

use crate::ai::Bot;
use crate::games::ataxx::AtaxxBoard;
use crate::uai::command::{Command, Position};

pub fn run(mut bot: impl Bot<AtaxxBoard>) -> std::io::Result<()> {
    let stdin = stdin();
    let mut handle = stdin.lock();
    let mut line = String::new();

    let mut curr_board = None;

    loop {
        line.clear();
        handle.read_line(&mut line)?;
        let line = line.trim();

        println!("info > {}", line);
        let command = Command::parse(line)
            .unwrap_or_else(|_| panic!("Failed to parse command '{}'", line));
        // println!("info received parsed {:?}", command);

        match command {
            Command::UAI => {
                println!("id name STTT-rs");
                println!("id author KarelPeeters");
                println!("uaiok");

                println!("info < id name STTT-rs");
                println!("info < id author KarelPeeters");
                println!("info < uaiok");
            }
            Command::IsReady => {
                println!("readyok");

                println!("info < readyok");
            }
            Command::NewGame => {
                curr_board = Some(AtaxxBoard::new_without_gaps());
            }
            Command::Position(position) => {
                curr_board = Some(match position {
                    Position::StartPos => AtaxxBoard::new_without_gaps(),
                    Position::Fen(fen) => AtaxxBoard::from_fen(fen),
                })
            }
            Command::Go(_) => {
                let curr_board = curr_board.as_ref().unwrap();
                let best_move = bot.select_move(&curr_board);
                println!("bestmove {}", best_move.to_uai());
                println!("info < bestmove {}", best_move.to_uai());
            }
            Command::Quit => {
                //nothing to do here
            }
        }
    }
}
