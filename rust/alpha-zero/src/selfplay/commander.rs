use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::str;

use board_game::board::Board;
use crossbeam::channel::Sender;

use cuda_nn_eval::tester::check_cudnn;
use nn_graph::onnx::load_graph_from_onnx_path;
use nn_graph::optimizer::{optimize_graph, OptimizerSettings};

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

                let loaded_graph = load_graph_from_onnx_path(path);
                let graph = optimize_graph(&loaded_graph, OptimizerSettings::default());

                let check_data = std::fs::read(path_bin)
                    .expect("Failed to read check data");
                check_cudnn(&graph, &check_data, false);
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

//TODO some proper error handling so we don't have to constantly restart the server
// and it's unclear whether it really crashed
pub fn read_command(reader: &mut BufReader<impl Read>) -> Command {
    let mut buffer = vec![];
    reader.read_until(b'\n', &mut buffer).unwrap();
    buffer.pop();

    let str = str::from_utf8(&buffer).unwrap();
    println!("Received command {}", str);

    serde_json::from_str::<Command>(str).unwrap()
}
