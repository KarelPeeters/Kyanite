use std::io;
use std::io::{stdin, Write};
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use regex::Regex;

use sttt::board::{Board, board_from_compact_string, board_to_compact_string, Coord};
use sttt::bot_game;
use sttt::bot_game::{RandomBot, Bot};
use sttt::minimax::move_minimax;
use sttt::mcts::MCTSBot;

fn main() {
    // _console_game();
    _bot_game();
    // _test_compact_string();
    // _time_mcts()

}

fn _time_mcts() {
    let mut board = Board::new();
    board.play(Coord::of_oo(4, 4));
    board.play(Coord::of_oo(4, 0));

    time(|| {
        MCTSBot::new(10_000, SmallRng::from_entropy()).play(&board);
    })
}

fn _test_compact_string() {
    let seed: [u8; 16] = SmallRng::from_entropy().gen();
    print!("Seed: {:?}", seed);

    let mut rand = SmallRng::from_seed(seed);

    loop {
        let mut board = Board::new();

        while let Some(mv) = board.random_available_move(&mut rand) {
            board.play(mv);

            let compact_string = board_to_compact_string(&board);
            let rev_board = board_from_compact_string(&compact_string);

            // print!("Board:\n{}\n{:#?}\nRev Board:\n{}\n{:#?}", board, board, rev_board, rev_board);
            assert_eq!(rev_board, board);

            println!("{}", compact_string);
        }
    }
}

fn _test_mm() {
    let board = Board::new();

    let start = Instant::now();
    let mv = move_minimax(&board, 10);
    println!("{:?}", mv);
    println!("{}", start.elapsed().as_millis() as f64 / 1000.0);
}

fn _follow_playout() {
    let moves = [35, 73, 9, 8, 77, 53, 76, 40, 39, 29, 20, 19, 11, 24, 59, 45, 2, 22, 37, 15, 58, 43, 67, 42, 54, 4, 41, 50, 47, 25, 70, 64, 17, 78, 57, 30, 34, 65, 3, 33, 44, 74, 1, 12, 28, 10, 13, 36, 0, 52, 68, 49, 38, 32, 31, ];

    let mut board = Board::new();
    for &mv in moves.iter() {
        board.play(Coord::of_o(mv));
        println!("{}", board);
    }
}

fn _bot_game() {
    let res = bot_game::run(
        || RandomBot,
        || MCTSBot::new(1000, SmallRng::from_entropy()),
        100,
        true,
    );

    println!("{:?}", res);
}

fn _console_game<B: Bot>(mut bot: B) {
    let move_regex = Regex::new(r"^(?P<om>\d+)\s*(?:,\s*)?(?P<os>\d+)$").unwrap();

    let mut history = Vec::new();
    let mut board = Board::new();

    println!("{}", board);

    let mut line = String::new();

    loop {
        //Player move
        'playerMove: loop {
            print!("Play move: ");
            io::stdout().flush().expect("Could not flush stdout");

            line.clear();
            stdin().read_line(&mut line).unwrap();
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
                    let mv = Coord::of_oo(om, os);
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
            println!("You won :)");
            break;
        }

        //Bot move
        let mv = bot.play(&board)
            .expect("Bot should return move");

        board.play(mv);
        println!("{}", board);

        if board.is_done() {
            println!("You lost :(");
            break;
        }
    }
}

#[allow(unused)]
fn time<R, F: FnOnce() -> R>(block: F) -> R {
    let start = Instant::now();
    let result = block();
    print!("Took {:02}s", (Instant::now() - start).as_secs_f32());
    result
}