use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};

use board_game::board::Board;
use board_game::games::ataxx::AtaxxBoard;
use board_game::games::chess::{ChessBoard, Rules};
use crossbeam::channel;

use cuda_sys::wrapper::handle::Device;

use crate::mapping::ataxx::AtaxxStdMapper;
use crate::mapping::BoardMapper;
use crate::mapping::chess::ChessStdMapper;
use crate::selfplay::collector::collector_main;
use crate::selfplay::commander::{commander_main, read_command};
use crate::selfplay::generator::generator_main;
use crate::selfplay::protocol::{Command, StartupSettings};

pub fn selfplay_server_main() {
    let (stream, addr) = TcpListener::bind("::1:63105").unwrap()
        .accept().unwrap();
    println!("Accepted connection {:?} on {:?}", stream, addr);

    let writer = BufWriter::new(&stream);
    let mut reader = BufReader::new(&stream);

    let startup_settings = wait_for_startup_settings(&mut reader);
    println!("Received startup settings:\n{:#?}", startup_settings);

    match &*startup_settings.game {
        "ataxx" => {
            selfplay_start(
                startup_settings,
                AtaxxBoard::default,
                AtaxxStdMapper,
                reader, writer,
            )
        }
        "chess" => {
            selfplay_start(
                startup_settings,
                || ChessBoard::default(),
                ChessStdMapper,
                reader, writer,
            )
        }
        game => {
            panic!("Unknown game '{}'", game);
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
    startup: StartupSettings,
    start_pos: impl Fn() -> B + Sync,
    mapper: impl BoardMapper<B>,
    reader: BufReader<impl Read>,
    writer: BufWriter<impl Write + Send>,
) {
    let mut cmd_senders = vec![];
    let (update_sender, update_receiver) = channel::unbounded();

    crossbeam::scope(|s| {
        for device in Device::all() {
            for _ in 0..startup.threads_per_device {
                let (cmd_sender, cmd_receiver) = channel::unbounded();
                cmd_senders.push(cmd_sender);
                let update_sender = update_sender.clone();

                let start_pos = &start_pos;
                let batch_size = startup.batch_size;
                s.spawn(move |_| {
                    generator_main(mapper, start_pos, device, batch_size, cmd_receiver, update_sender)
                });
            }
        }

        s.spawn(|_| {
            collector_main(writer, startup.games_per_gen, startup.first_gen, &startup.output_folder, mapper, update_receiver)
        });

        commander_main(reader, cmd_senders, update_sender);
    }).unwrap();
}
