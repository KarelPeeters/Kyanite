use anyhow::Result;
use futures_util::TryStreamExt;
use licorice::client::Lichess;
use licorice::models::board::BoardState;
use shakmaty::san::San;
use shakmaty::uci::Uci;
use shakmaty::{Chess, Position, Setup};
use std::io;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "playfish", about = "Play against the Stockfish AI at lichess")]
struct Opt {
    #[structopt(short, long)]
    token: String,

    #[structopt(short, long, default_value = "1")]
    #[structopt(possible_values = &["1", "2", "3", "4", "5", "6", "7", "8"])]
    level: u8,

    #[structopt(short, long, default_value = "white")]
    #[structopt(possible_values = &["white", "black"])]
    color: String,
}

fn print_oriented(pos: &Chess, is_pov_white: bool) {
    let ascii_board = format!("{:?}", pos.board());
    let mut oriented: Vec<String> = vec![];
    let mut start = 0;
    let esc = 27 as char;

    if is_pov_white {
        for rank in ascii_board.split_terminator('\n') {
            let new_rank = format!("{}  {esc}[1m{}{esc}[0m", rank, 8 - start, esc = esc);
            start += 1;
            oriented.push(new_rank);
        }
        println!(
            "{esc}[2J{esc}[1;1H{}\n{esc}[1m{}{esc}[0m\n\n\n",
            oriented.join("\n"),
            "a b c d e f g h".to_owned(),
            esc = esc
        )
    } else {
        for rank in ascii_board.split_terminator('\n').rev() {
            start += 1;
            let new_rank = format!(
                "{}  {esc}[1m{}{esc}[0m",
                rank.chars().rev().collect::<String>(),
                start,
                esc = esc
            );
            oriented.push(new_rank);
        }
        println!(
            "{esc}[2J{esc}[1;1H{}\n{esc}[1m{}{esc}[0m\n\n\n",
            oriented.join("\n"),
            "H G F E D C B A".to_owned(),
            esc = esc
        )
    }
}

fn user_move(pos: &mut Chess, is_pov_white: bool) -> Result<(Uci, bool)> {
    let esc = 27 as char;
    loop {
        println!("Waiting for YOU to move...");
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let my_move = &input[..input.len() - 1];
        match my_move.parse::<San>() {
            Ok(san) => match san.to_move(pos) {
                Ok(m) => {
                    pos.play_unchecked(&m);
                    print_oriented(&*pos, is_pov_white);
                    println!("{}[2FYou have played {}\n", 27 as char, m);
                    let uci = Uci::from_standard(&m);
                    if pos.is_game_over() {
                        if pos.is_checkmate() {
                            println!("You have slain the fish :tada:")
                        } else {
                            println!("It was an even battle :neutral:")
                        }
                        return Ok((uci, true));
                    } else {
                        return Ok((uci, false));
                    }
                }
                Err(e) => {
                    println!("{esc}[1F{esc}[K{esc}[2F{}{esc}[K {}", e, my_move, esc = esc);
                    continue;
                }
            },
            Err(e) => {
                println!("{esc}[1F{esc}[K{esc}[2F{esc}[K{} {}", e, my_move, esc = esc);
                continue;
            }
        }
    }
}

fn fish_move(moves: String, pos: &mut Chess, is_pov_white: bool) -> Result<bool> {
    let new_move = moves.rsplitn(2, ' ').next().unwrap();
    let uci: Uci = new_move.parse()?;
    let m = uci.to_move(pos)?;
    pos.play_unchecked(&m);
    print_oriented(&*pos, is_pov_white);
    println!("{}[2FFish has played {}\n", 27 as char, m);
    if pos.is_game_over() {
        if pos.is_checkmate() {
            println!("You have been slain by the fish :sadge:")
        } else {
            println!("It was an even battle :neutral:")
        }
        return Ok(true);
    }
    Ok(false)
}

#[tokio::main]
async fn main() -> Result<()> {
    let opt = Opt::from_args();
    let lichess = Lichess::new(opt.token);
    let mut pos = Chess::default();

    let mut your_turn = true;
    let mut is_pov_white = false;

    let game_id = lichess
        .challenge_stockfish(opt.level, Some(&[("color", &opt.color)]))
        .await?
        .id;
    let mut board_stream = lichess.stream_board_game_state(&game_id).await?;

    let first = board_stream.try_next().await?;
    if opt.color == "white" {
        is_pov_white = true;
        print_oriented(&pos, is_pov_white);
        let (uci, _) = user_move(&mut pos, is_pov_white)?;
        lichess
            .make_a_board_move(&game_id, &uci.to_string(), false)
            .await?;
        println!("Waiting for FISH to move...");
        board_stream.try_next().await?;
        your_turn = false;
    } else if let Some(BoardState::GameFull(game_full)) = first {
        fish_move(game_full.state.moves, &mut pos, is_pov_white)?;
    }

    loop {
        if your_turn {
            let (uci, done) = user_move(&mut pos, is_pov_white)?;
            lichess
                .make_a_board_move(&game_id, &uci.to_string(), false)
                .await?;
            if done {
                break;
            } else {
                println!("Waiting for FISH to move...");
                board_stream.try_next().await?;
            }
        } else if let Some(BoardState::GameState(game_state)) = board_stream.try_next().await? {
            if fish_move(game_state.moves, &mut pos, is_pov_white)? {
                break;
            }
        }
        your_turn = !your_turn;
    }
    Ok(())
}
