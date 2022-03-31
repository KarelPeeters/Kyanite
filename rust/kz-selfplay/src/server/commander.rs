use std::io::{BufRead, BufReader, Read};
use std::str;
use std::sync::Arc;

use board_game::board::Board;
use crossbeam::channel::Sender;

use kz_core::mapping::BoardMapper;
use kz_core::network::muzero::{MuZeroFusedGraphs, MuZeroGraphs};
use nn_graph::optimizer::OptimizerSettings;

use crate::server::protocol::{Command, GeneratorUpdate, Settings};

pub fn commander_main<B: Board, M: BoardMapper<B>>(
    mut reader: BufReader<impl Read>,
    mapper: M,
    settings_senders: Vec<Sender<Settings>>,
    graph_senders: Vec<Sender<Arc<MuZeroFusedGraphs<B, M>>>>,
    update_sender: Sender<GeneratorUpdate<B>>,
) {
    loop {
        let cmd = read_command(&mut reader);

        match cmd {
            Command::StartupSettings(_) => panic!("Already received startup settings"),
            Command::NewSettings(settings) => {
                for sender in &settings_senders {
                    sender.send(settings.clone()).unwrap();
                }
            }
            Command::NewNetwork(path) => {
                //TODO implement non-muzero support again

                println!("Commander loading & optimizing new network {:?}", path);
                let graphs = MuZeroGraphs::load(&path, mapper);
                let graphs = graphs.optimize(OptimizerSettings::default());
                let fused = graphs.fuse(OptimizerSettings::default());

                // put it in an arc so we don't need to clone it a bunch of times
                let fused = Arc::new(fused);

                println!("Sending new network to executors");
                for sender in &graph_senders {
                    sender.send(Arc::clone(&fused)).unwrap();
                }
            }
            Command::WaitForNewNetwork => {
                //TODO implement this again, maybe only send to the relevant GPU?
                eprintln!("Waiting for new network is currently not implemented");
            }
            Command::Stop => {
                //TODO this is probably not enoug any more, we need to stop the gpu executors, cpu threads and rebatchers as well
                update_sender.send(GeneratorUpdate::Stop).unwrap();
                break;
            }
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
