use std::fs::File;
use std::io::{BufRead, stdin};
use std::io::Write;

use itertools::Itertools;

use crate::board::{Board, Player};
use crate::games::ataxx::{AtaxxBoard, Move};
use crate::uai::command::{Command, Position};

pub fn run(mut bot: impl FnMut(&AtaxxBoard, u32) -> (Move, u64)) -> std::io::Result<()> {
    //warmup
    bot(&AtaxxBoard::new_without_gaps(), 1000);

    let stdin = stdin();
    let mut handle = stdin.lock();
    let mut line = String::new();

    let mut curr_board = None;

    let mut file = std::io::BufWriter::new(File::create("log.txt").unwrap());

    loop {
        line.clear();
        handle.read_line(&mut line)?;
        let line = line.trim();
        write!(&mut file, "{}\n", line).unwrap();
        file.flush().unwrap();

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
            Command::SetOption { name, value } => {
                println!("info < ignoring command setoption, name={}, value={}", name, value);
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
            Command::Go(derp) => {
                let times = derp.chars()
                    .filter(|c| c.is_ascii_digit() || c.is_whitespace())
                    .collect::<String>()
                    .split(" ")
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<u32>().unwrap())
                    .collect_vec();
                println!("{:?}", times);

                let curr_board = curr_board.as_ref().unwrap();

                let time_left = match curr_board.next_player() {
                    Player::A => times[0],
                    Player::B => times[1],
                };

                let (best_move, nodes) = bot(curr_board, time_left);
                write!(&mut file, "nodes: {}", nodes).unwrap();
                file.flush().unwrap();
                println!("bestmove {}", best_move.to_uai());
                println!("info < bestmove {}", best_move.to_uai());
            }
            Command::Quit => {
                //nothing to do here
            }
        }
    }
}
