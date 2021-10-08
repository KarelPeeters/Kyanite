use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::str;

use board_game::board::Board;
use crossbeam::channel::Sender;

use cuda_nn_eval::tester::check_cudnn;
use nn_graph::onnx::load_graph_from_onnx_path;

use crate::selfplay::protocol::{Command, GeneratorUpdate};

pub fn commander_main<B: Board>(
    mut reader: BufReader<impl Read>,
    cmd_senders: Vec<Sender<Command>>,
    update_sender: Sender<GeneratorUpdate<B>>,
) {
    loop {
        let cmd = read_command(&mut reader);

        if let Command::NewNetwork(path) = &cmd {
            let path_bin = PathBuf::from(path).with_extension("bin");
            if path_bin.exists() {
                println!("Commander checking new network {}", path);

                let graph = load_graph_from_onnx_path(path);
                let check_data = std::fs::read(path_bin)
                    .expect("Failed to read check data");
                check_cudnn(&graph, &check_data);
            }
        }

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
