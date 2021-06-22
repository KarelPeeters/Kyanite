use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Instant;

use derive_more::From;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use sttt::board::{board_from_compact_string, Coord};
use sttt::bot_game::Bot;
use sttt::mcts::MCTSBot;

fn main() -> Result<(), Error> {
    server_loop(MCTSBot::new(10_000, 2.0, SmallRng::from_entropy()))
}

#[derive(Debug, From)]
enum Error {
    IO(std::io::Error),
    Utf8(std::str::Utf8Error),
}

fn server_loop<B: Bot>(mut bot: B) -> Result<(), Error> {
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
