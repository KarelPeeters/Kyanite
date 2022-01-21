use std::fmt::{Display, Formatter};
use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};

use board_game::board::Board;
use board_game::games::ataxx::AtaxxBoard;
use board_game::games::chess::ChessBoard;
use board_game::games::sttt::STTTBoard;
use board_game::games::ttt::TTTBoard;
use crossbeam::channel;
use itertools::Itertools;

use cuda_nn_eval::Device;

use crate::mapping::ataxx::AtaxxStdMapper;
use crate::mapping::BoardMapper;
use crate::mapping::chess::ChessStdMapper;
use crate::mapping::sttt::STTTStdMapper;
use crate::mapping::ttt::TTTStdMapper;
use crate::selfplay::collector::collector_main;
use crate::selfplay::commander::{commander_main, read_command};
use crate::selfplay::generator::generator_main;
use crate::selfplay::protocol::{Command, StartupSettings};

#[derive(Debug, Copy, Clone)]
enum Game {
    TTT,
    STTT,
    Chess,
    Ataxx { size: u8 },
}

pub fn selfplay_server_main() {
    let (stream, addr) = TcpListener::bind("127.0.0.1:63105").unwrap()
        .accept().unwrap();
    println!("Accepted connection {:?} on {:?}", stream, addr);

    let writer = BufWriter::new(&stream);
    let mut reader = BufReader::new(&stream);

    let startup_settings = wait_for_startup_settings(&mut reader);
    println!("Received startup settings:\n{:#?}", startup_settings);

    let game = Game::parse(&startup_settings.game)
        .unwrap_or_else(|| panic!("Unknown game '{}'", startup_settings.game));

    match game {
        Game::TTT => {
            selfplay_start(
                game,
                startup_settings,
                TTTBoard::default,
                TTTStdMapper,
                reader, writer,
            )
        }
        Game::STTT => {
            selfplay_start(
                game,
                startup_settings,
                STTTBoard::default,
                STTTStdMapper,
                reader, writer,
            )
        }
        Game::Ataxx { size } => {
            selfplay_start(
                game,
                startup_settings,
                || AtaxxBoard::diagonal(size),
                AtaxxStdMapper::new(size),
                reader, writer,
            )
        }
        Game::Chess => {
            selfplay_start(
                game,
                startup_settings,
                ChessBoard::default,
                ChessStdMapper,
                reader, writer,
            )
        }
    }
}

fn wait_for_startup_settings(reader: &mut BufReader<&TcpStream>) -> StartupSettings {
    match read_command(reader) {
        Command::StartupSettings(startup) =>
            startup,
        command =>
            panic!("Must receive startup settings before any other command, got {:?}", command),
    }
}

fn selfplay_start<B: Board>(
    game: Game,
    startup: StartupSettings,
    start_pos: impl Fn() -> B + Sync,
    mapper: impl BoardMapper<B>,
    reader: BufReader<impl Read>,
    writer: BufWriter<impl Write + Send>,
) {
    let mut cmd_senders = vec![];
    let (update_sender, update_receiver) = channel::unbounded();

    crossbeam::scope(|s| {
        let devices = Device::all().collect_vec();
        let thread_count = devices.len() * startup.threads_per_device;

        for device in devices {
            for thread_id in 0..startup.threads_per_device {
                let (cmd_sender, cmd_receiver) = channel::unbounded();
                cmd_senders.push(cmd_sender);
                let update_sender = update_sender.clone();

                let start_pos = &start_pos;
                let batch_size = startup.batch_size;
                s.builder().name(format!("generator-d{}-{}", device.inner(), thread_id)).spawn(move |_| {
                    generator_main(thread_id, mapper, start_pos, device, batch_size, cmd_receiver, update_sender)
                }).unwrap();
            }
        }

        s.builder().name("collector".to_string()).spawn(move |_| {
            collector_main(
                &game.to_string(),
                writer,
                startup.games_per_gen,
                startup.first_gen,
                &startup.output_folder,
                mapper,
                update_receiver,
                thread_count,
                startup.reorder_games,
            )
        }).unwrap();

        commander_main(reader, cmd_senders, update_sender);
    }).unwrap();
}

impl Game {
    fn parse(str: &str) -> Option<Game> {
        match str {
            "ttt" => return Some(Game::TTT),
            "sttt" => return Some(Game::STTT),
            "chess" => return Some(Game::Chess),
            "ataxx" => return Some(Game::Ataxx { size: 7 }),
            _ => {}
        };

        let start = "ataxx-";
        if let Some(size) = str.strip_prefix(start) {
            let size: u8 = size.parse().ok()?;
            return Some(Game::Ataxx { size });
        }

        None
    }
}

impl Display for Game {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Game::TTT => write!(f, "ttt"),
            Game::STTT => write!(f, "sttt"),
            Game::Chess => write!(f, "chess"),
            Game::Ataxx { size } => write!(f, "ataxx-{}", size),
        }
    }
}