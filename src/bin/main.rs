use std::io;
use std::io::{Read, stdin, Write};
use std::net::TcpListener;
use std::time::Instant;

use derive_more::From;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use regex::Regex;

use sttt::board::{Board, board_from_compact_string, board_to_compact_string, Coord};
use sttt::bot_game;
use sttt::bot_game::{Bot, RandomBot};
use sttt::mcts::{MCTSBot, Rand};
use sttt::minimax::MiniMaxBot;

//data is organized as a flattened list of (upper, result)
// upper == 0 means a bool was requested
struct ReplayRandom<'d> {
    data: &'d [u32],
    index: usize,
}

impl<'d> ReplayRandom<'d> {
    fn new(data: &'d [u32]) -> Self {
        assert_eq!(data.len() % 2, 0);
        ReplayRandom { data, index: 0 }
    }
}

impl Rand for ReplayRandom<'_> {
    fn gen_range(&mut self, low: usize, high: usize) -> usize {
        assert_eq!(low, 0);

        let expect_high = self.data[self.index] as usize;
        let result = self.data[self.index+1] as usize;
        self.index += 2;

        if expect_high != high {
            panic!("Failed at index {}, expected {}, got {}", self.index, expect_high, high)
        }

        return result;
    }

    fn gen(&mut self) -> bool {
        let value = self.gen_range(0, 0);
        assert!(value == 1 || value == 0);
        return value != 0;
    }
}

struct RecordRandom<R: Rng> {
    inner: R,
    result: Vec<u32>,
}

impl<R: Rng> Rand for RecordRandom<R> {
    fn gen_range(&mut self, low: usize, high: usize) -> usize {
        assert_eq!(low, 0);
        let value = self.inner.gen_range(low, high);

        self.result.push(high as u32);
        self.result.push(value as u32);

        value
    }

    fn gen(&mut self) -> bool {
        let value = self.inner.gen();

        self.result.push(0);
        self.result.push(value as u32);

        value
    }
}

impl<R: Rng> Drop for RecordRandom<R> {
    fn drop(&mut self) {
        println!("{:?}", self.result)
    }
}

fn main() {
    // _console_game();
    // _bot_game();
    // _test_compact_string();
    // _time_mcts()

    // println!("Main");

    _bot_server(MCTSBot::new(100_000, SmallRng::from_entropy())).unwrap()

    // let recording = [8, 5, 9, 1, 9, 8, 9, 5, 9, 3, 9, 8, 8, 3, 8, 6, 8, 3, 7, 5, 9, 8, 7, 3, 6, 1, 8, 2, 9, 2, 8, 6, 8, 1, 7, 3, 5, 2, 7, 2, 7, 6, 6, 5, 5, 3, 7, 2, 6, 1, 6, 4, 6, 2, 8, 2, 6, 3, 4, 2, 7, 2, 3, 2, 4, 3, 7, 2, 5, 2, 2, 1, 5, 1, 37, 1, 36, 1, 4, 1, 4, 2, 6, 5, 3, 0, 7, 5, 6, 3, 5, 3, 5, 2, 4, 3, 24, 5, 23, 5, 5, 0, 4, 2, 4, 1, 19, 5, 3, 2, 3, 0, 2, 0, 2, 1, 5, 2, 11, 4, 7, 6, 9, 0, 9, 3, 9, 2, 9, 7, 9, 0, 8, 1, 9, 8, 8, 0, 8, 2, 8, 5, 9, 1, 7, 1, 6, 0, 7, 4, 9, 7, 8, 4, 8, 6, 7, 4, 8, 2, 7, 4, 7, 1, 5, 4, 6, 1, 6, 0, 6, 1, 5, 3, 7, 6, 7, 4, 6, 1, 4, 3, 5, 1, 8, 4, 7, 5, 4, 1, 6, 1, 4, 3, 6, 3, 6, 0, 5, 0, 4, 0, 5, 3, 3, 1, 5, 3, 4, 3, 5, 4, 4, 0, 26, 10, 2, 1, 1, 0, 3, 1, 5, 0, 3, 0, 6, 0, 13, 7, 12, 0, 6, 2, 9, 7, 9, 4, 7, 1, 9, 7, 8, 7, 9, 1, 8, 0, 9, 7, 7, 0, 8, 4, 6, 3, 9, 0, 7, 0, 6, 0, 7, 6, 8, 2, 9, 0, 57, 17, 5, 0, 55, 6, 54, 17, 8, 0, 6, 2, 6, 4, 6, 5, 5, 3, 9, 4, 4, 3, 7, 1, 7, 5, 7, 1, 5, 4, 6, 5, 5, 2, 8, 5, 6, 4, 4, 1, 6, 3, 3, 1, 7, 5, 3, 2, 5, 2, 6, 5, 4, 3, 28, 27, 4, 3, 2, 1, 2, 1, 23, 22, 2, 0, 5, 1, 4, 0, 5, 4, 13, 11, 4, 1, 3, 0, 3, 1, 5, 1, 4, 2, 1, 0, 0, 0, 5, 2, 9, 0, 9, 8, 9, 2, 9, 5, 9, 3, 8, 4, 8, 3, 7, 4, 9, 0, 8, 6, 8, 5, 7, 3, 6, 2, 8, 4, 5, 0, 7, 0, 6, 3, 4, 1, 7, 4, 6, 4, 9, 4, 3, 2, 8, 5, 5, 1, 7, 0, 5, 4, 8, 7, 7, 1, 9, 0, 46, 43, 6, 0, 39, 18, 4, 0, 8, 3, 2, 0, 7, 5, 7, 6, 6, 3, 6, 0, 6, 4, 3, 1, 28, 26, 5, 3, 26, 14, 4, 3, 2, 1, 4, 2, 16, 5, 5, 3, 14, 4, 4, 0, 4, 0, 3, 0, 4, 2, 9, 8, 9, 1, 9, 5, 8, 5, 7, 5, 9, 1, 8, 5, 8, 3, 7, 4, 7, 3, 6, 0, 9, 0, 8, 6, 9, 0, 7, 3, 6, 3, 9, 8, 8, 0, 6, 3, 5, 2, 8, 4, 5, 1, 7, 4, 4, 3, 7, 5, 8, 3, 3, 1, 9, 7, 7, 2, 7, 4, 4, 3, 6, 2, 42, 30, 6, 5, 5, 3, 39, 35, 8, 5, 34, 19, 6, 3, 5, 0, 5, 0, 21, 13, 4, 2, 15, 9, 5, 4, 13, 0, 12, 0, 11, 5, 5, 4, 8, 3, 7, 2, 6, 4, 5, 1, 3, 2, 9, 1, 9, 6, 9, 6, 8, 1, 8, 7, 9, 5, 9, 8, 8, 0, 9, 3, 9, 7, 8, 5, 7, 1, 9, 3, 8, 1, 7, 1, 6, 5, 7, 5, 6, 4, 8, 2, 8, 2, 7, 6, 7, 1, 6, 4, 6, 4, 5, 2, 7, 3, 7, 3, 46, 13, 8, 5, 5, 4, 6, 0, 5, 1, 4, 1, 6, 3, 7, 2, 38, 36, 4, 3, 4, 3, 29, 24, 28, 10, 5, 3, 22, 7, 7, 2, 3, 1, 19, 8, 2, 0, 16, 6, 15, 6, 6, 3, 13, 2, 4, 0, 11, 0, 3, 2, 7, 1, 4, 0, 5, 0, 4, 3, 3, 2, 1, 0, 0, 0, 2, 0, 9, 4, 7, 3, 9, 8, 9, 2, 9, 2, 8, 2, 9, 4, 6, 2, 8, 5, 9, 2, 7, 6, 8, 0, 8, 0, 7, 6, 7, 1, 7, 6, 6, 0, 9, 3, 6, 0, 6, 5, 9, 0, 5, 0, 8, 7, 5, 1, 8, 4, 5, 0, 7, 3, 4, 1, 8, 3, 47, 15, 8, 5, 7, 0, 4, 2, 7, 1, 6, 2, 5, 3, 6, 3, 5, 0, 3, 1, 5, 2, 4, 2, 4, 0, 4, 1, 5, 0, 27, 19, 26, 9, 6, 0, 4, 1, 23, 12, 5, 4, 4, 2, 4, 3, 3, 0, 3, 0, 15, 4, 5, 0, 3, 0, 2, 0, 6, 3, 3, 1, 2, 1, 1, 0, 1, 0, 9, 4, 7, 0, 9, 5, 9, 6, 9, 8, 9, 1, 8, 3, 9, 5, 8, 0, 8, 6, 9, 6, 8, 5, 7, 3, 6, 4, 8, 7, 8, 3, 5, 0, 9, 6, 7, 0, 7, 0, 6, 5, 7, 5, 7, 5, 6, 5, 49, 16, 4, 1, 5, 2, 8, 6, 6, 2, 7, 3, 7, 2, 6, 2, 5, 1, 7, 2, 4, 0, 5, 0, 6, 2, 4, 1, 3, 1, 22, 8, 6, 0, 20, 12, 19, 13, 18, 0, 14, 9, 9, 8, 8, 1, 7, 2, 3, 1, 9, 1, 9, 6, 9, 0, 8, 0, 7, 2, 7, 3, 9, 6, 8, 6, 9, 7, 8, 0, 6, 1, 9, 0, 5, 3, 7, 1, 9, 8, 9, 2, 8, 3, 8, 0, 8, 3, 7, 2, 6, 1, 7, 0, 53, 40, 6, 4, 5, 3, 7, 2, 44, 24, 4, 1, 5, 2, 6, 2, 36, 15, 6, 2, 34, 20, 6, 3, 32, 15, 4, 1, 7, 3, 29, 28, 7, 3, 27, 2, 5, 1, 4, 0, 5, 1, 4, 2, 3, 2, 6, 4, 4, 3, 5, 3, 3, 1, 11, 5, 8, 0, 7, 2, 6, 4, 3, 1, 1, 0, 0, 1, 8, 3, 7, 2, 9, 0, 8, 5, 9, 0, 7, 5, 9, 3, 8, 2, 7, 0, 9, 4, 6, 1, 9, 1, 8, 5, 8, 5, 7, 2, 6, 3, 6, 1, 8, 6, 8, 2, 7, 5, 5, 4, 9, 0, 6, 4, 9, 3, 5, 1, 5, 4, 8, 7, 7, 3, 4, 0, 7, 6, 6, 1, 6, 5, 5, 4, 7, 6, 4, 3, 4, 0, 6, 1, 5, 1, 33, 0, 4, 2, 4, 2, 6, 2, 3, 1, 3, 1, 8, 7, 21, 12, 2, 0, 18, 11, 5, 2, 16, 2, 13, 3, 5, 1, 3, 2, 4, 1, 2, 0, 8, 0, 7, 0, 6, 3, 2, 1, 3, 2, 1, 0, 9, 4, 7, 6, 9, 8, 8, 0, 9, 4, 6, 3, 9, 7, 8, 5, 9, 0, 8, 4, 8, 4, 7, 2, 8, 7, 7, 6, 6, 4, 9, 0, 7, 1, 9, 5, 6, 2, 5, 4, 8, 4, 5, 3, 7, 5, 7, 5, 6, 5, 45, 22, 5, 3, 4, 0, 8, 7, 4, 3, 40, 10, 39, 28, 6, 4, 5, 1, 8, 3, 7, 5, 2, 0, 7, 6, 32, 2, 6, 0, 5, 1, 6, 5, 4, 2, 22, 5, 5, 0, 4, 1, 3, 2, 3, 0, 13, 5, 4, 3, 2, 0, 10, 1, 9, 6, 3, 2, 7, 0, 5, 4, 2, 0, 3, 2, 2, 1, 1, 0, 0, 1, 9, 8, 9, 6, 9, 5, 9, 0, 9, 8, 8, 0, 8, 6, 8, 7, 7, 2, 9, 3, 8, 3, 7, 0, 7, 3, 7, 1, 9, 1, 8, 5, 7, 5, 6, 5, 8, 7, 7, 3, 6, 4, 6, 4, 8, 2, 5, 1, 9, 7, 5, 0, 6, 0, 47, 11, 6, 5, 6, 3, 7, 2, 5, 0, 7, 3, 4, 3, 5, 0, 39, 0, 38, 32, 4, 3, 3, 1, 4, 2, 3, 0, 5, 4, 5, 2, 2, 0, 7, 4, 6, 3, 2, 0, 6, 4, 21, 5, 4, 2, 5, 4, 4, 1, 4, 3, 3, 1, 15, 7, 4, 1, 3, 1, 10, 8, 3, 0, 7, 6, 6, 1, 9, 3, 9, 6, 8, 3, 7, 5, 9, 5, 9, 1, 9, 1, 8, 6, 8, 6, 7, 3, 8, 0, 9, 6, 7, 3, 8, 7, 9, 6, 58, 39, 7, 2, 6, 4, 6, 2, 9, 1, 7, 5, 52, 34, 8, 0, 8, 6, 5, 2, 6, 0, 7, 5, 5, 4, 4, 2, 41, 21, 5, 2, 4, 2, 4, 2, 3, 1, 4, 1, 7, 4, 32, 7, 6, 4, 30, 18, 8, 7, 7, 2, 5, 2, 3, 0, 5, 1, 2, 0, 4, 3, 6, 3, 9, 1, 9, 4, 7, 5, 9, 3, 9, 4, 6, 2, 9, 7, 8, 1, 8, 7, 9, 3, 8, 7, 8, 3, 5, 1, 7, 2, 8, 7, 7, 0, 9, 3, 7, 4, 8, 1, 7, 5, 7, 5, 7, 6, 6, 1, 6, 5, 9, 1, 6, 1, 5, 4, 6, 4, 8, 0, 8, 0, 7, 2, 44, 27, 5, 2, 4, 0, 6, 3, 6, 4, 5, 1, 3, 2, 37, 8, 5, 4, 5, 3, 4, 3, 27, 23, 26, 18, 3, 0, 5, 2, 23, 4, 2, 1, 20, 12, 4, 1, 2, 0, 14, 6, 1, 0, 5, 1, 11, 5, 4, 0, 3, 1, 2, 1, 7, 1, 4, 3, 9, 5, 9, 7, 9, 6, 9, 7, 8, 2, 9, 6, 8, 4, 7, 2, 8, 7, 9, 0, 9, 7, 7, 2, 8, 7, 8, 6, 6, 0, 8, 0, 7, 1, 7, 6, 5, 4, 7, 2, 7, 3, 6, 3, 6, 0, 6, 5, 6, 5, 5, 0, 9, 8, 4, 3, 7, 2, 6, 5, 8, 5, 7, 4, 5, 3, 4, 3, 3, 2, 6, 4, 6, 5, 37, 32, 5, 1, 5, 1, 8, 1, 7, 0, 4, 1, 4, 3, 30, 7, 4, 1, 4, 3, 27, 13, 3, 1, 5, 0, 22, 20, 4, 1, 5, 2, 4, 1, 5, 0, 3, 2];
    // let rand = ReplayRandom::new(&recording);

    // let mut bot: MCTSBot<_> = MCTSBot::new(15, SmallRng::from_entropy());
    // let mut board = Board::new();
    // board.play(Coord::of_oo(4, 4));
    // let mv = bot.play(&board);
    // println!("returned move: {:?}", mv.map(|mv| mv.o()));
}

#[derive(Debug, From)]
enum Error {
    IO(std::io::Error),
    Utf8(std::str::Utf8Error),
}

fn _bot_server<B: Bot>(mut bot: B) -> Result<(), Error> {
    println!("Before bind");

    let listener = TcpListener::bind("::1:1576")?;

    println!("Waiting for connection");
    for stream in listener.incoming() {
        println!("Got stream");

        let mut stream = stream?;

        loop {
            println!("Listening");
            let mut buf = [0; 81];
            stream.read_exact(&mut buf)?;

            let string = std::str::from_utf8(&buf)?;
            println!("Received board {:?}", string);

            let board = board_from_compact_string(string);

            let start = Instant::now();
            let mv = bot.play(&board);
            println!("Bot took {}s to find move", (Instant::now() - start).as_secs_f32());

            println!("Replying move {:?}", mv);

            let mv_int = mv.map(Coord::o).unwrap_or(100);
            stream.write(&[mv_int])?;

            println!("Reply done");
        }
    }

    Ok(())
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
    let seed: [u8; 16] = Rng::gen(&mut SmallRng::from_entropy());
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
    let mv = MiniMaxBot::new(10).play(&board);
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