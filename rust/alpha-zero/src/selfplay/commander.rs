use std::io::{BufRead, BufReader, Read};
use std::str;

use crossbeam::channel::Sender;

use board_game::board::Board;

use crate::selfplay::protocol::{Command, GeneratorUpdate};

pub fn commander_main<B: Board>(
    mut reader: BufReader<impl Read>,
    cmd_senders: Vec<Sender<Command>>,
    update_sender: Sender<GeneratorUpdate<B>>,
) {
    loop {
        let cmd = read_command(&mut reader);

        for s in &cmd_senders {
            s.send(cmd.clone()).unwrap();
        }

        if let Command::Stop = cmd {
            update_sender.send(GeneratorUpdate::Stop).unwrap();
            break;
        }
    }
}

pub fn read_command(reader: &mut BufReader<impl Read>) -> Command {
    let mut buffer = vec![];
    reader.read_until(b'\n', &mut buffer).unwrap();
    buffer.pop();

    let str = str::from_utf8(&buffer).unwrap();
    println!("Received command {}", str);

    serde_json::from_str::<Command>(str).unwrap()
}
