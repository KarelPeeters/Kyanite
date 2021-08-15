use std::io::{BufReader, BufWriter};
use std::net::{TcpListener, TcpStream};

use crossbeam::channel;

use board_game::board::Board;
use cuda_sys::wrapper::handle::Device;

use crate::network::Network;
use crate::selfplay::collector::collector_main;
use crate::selfplay::commander::{commander_main, read_command};
use crate::selfplay::core::Output;
use crate::selfplay::generator::generator_main;
use crate::selfplay::protocol::{Command, StartupSettings};

pub fn selfplay_server_main<B: Board, O: Output<B>, N: Network<B>>(
    game_name: &str,
    start_pos: impl Fn() -> B + Sync,
    output: impl Fn(&str) -> O + Send,
    load_network: impl Fn(String, usize, Device) -> N + Sync,
) {
    let (stream, addr) = TcpListener::bind("::1:63105").unwrap()
        .accept().unwrap();
    println!("Accepted connection {:?} on {:?}", stream, addr);
    selfplay_handle_connection(game_name, start_pos, output, load_network, stream);
}

fn wait_for_startup_settings(reader: &mut BufReader<&TcpStream>) -> StartupSettings {
    match read_command(reader) {
        Command::StartupSettings(startup) =>
            startup,
        command =>
            panic!("Must receive startup settings before any other command, got {:?}", command),
    }
}

fn selfplay_handle_connection<B: Board, O: Output<B>, N: Network<B>>(
    game_name: &str,
    start_pos: impl Fn() -> B + Sync,
    output: impl Fn(&str) -> O + Send,
    load_network: impl Fn(String, usize, Device) -> N + Sync,
    stream: TcpStream,
) {
    let writer = BufWriter::new(&stream);
    let mut reader = BufReader::new(&stream);

    let startup = wait_for_startup_settings(&mut reader);
    assert_eq!(game_name, startup.game);

    let mut cmd_senders = vec![];
    let (update_sender, update_receiver) = channel::unbounded();

    crossbeam::scope(|s| {
        for device in Device::all() {
            for _ in 0..startup.threads_per_device {
                let (cmd_sender, cmd_receiver) = channel::unbounded();
                cmd_senders.push(cmd_sender);
                let update_sender = update_sender.clone();

                let start_pos = &start_pos;
                let load_network = &load_network;
                let batch_size = startup.batch_size;
                s.spawn(move |_| {
                    generator_main(start_pos, load_network, device, batch_size, cmd_receiver, update_sender)
                });
            }
        }

        s.spawn(|_| {
            collector_main(writer, startup.games_per_file, &startup.output_folder, output, update_receiver)
        });

        commander_main(reader, cmd_senders, update_sender);
    }).unwrap();
}
