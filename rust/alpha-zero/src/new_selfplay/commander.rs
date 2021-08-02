use std::io::{BufRead, BufReader, Read};
use std::str;

use crossbeam::channel::Sender;

use board_game::board::Board;

use crate::new_selfplay::core::{Command, GeneratorUpdate};

pub fn commander_main<B: Board>(
    mut reader: BufReader<impl Read>,
    cmd_senders: Vec<Sender<Command>>,
    update_sender: Sender<GeneratorUpdate<B>>,
) {
    let mut buffer = vec![];

    loop {
        buffer.clear();
        reader.read_until(b'\n', &mut buffer).unwrap();

        let str = str::from_utf8(&buffer).unwrap();
        println!("Received command {}", str);

        let cmd = serde_json::from_str::<Command>(str).unwrap();

        for s in &cmd_senders {
            s.send(cmd.clone()).unwrap();
        }

        if let Command::Stop = cmd {
            update_sender.send(GeneratorUpdate::Stop).unwrap();
            break;
        }
    }
}
