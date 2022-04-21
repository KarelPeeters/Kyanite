use std::io::{BufRead, BufReader, Read};
use std::str;
use std::sync::Arc;

use board_game::board::Board;
use flume::Sender;

use crate::server::protocol::{Command, GeneratorUpdate, Settings};
use crate::server::server::GraphSender;

pub fn commander_main<B: Board, G>(
    mut reader: BufReader<impl Read>,
    settings_senders: Vec<Sender<Settings>>,
    graph_senders: Vec<GraphSender<G>>,
    update_sender: Sender<GeneratorUpdate<B>>,
    load_graph: impl Fn(&str) -> G,
) {
    loop {
        let cmd = read_command(&mut reader);

        match cmd {
            Command::StartupSettings(_) => panic!("Already received startup settings"),
            Command::NewSettings(settings) => {
                assert!(
                    !settings.random_symmetries,
                    "Random symmetries are currently not supported"
                );

                for sender in &settings_senders {
                    sender.send(settings.clone()).unwrap();
                }
            }
            Command::NewNetwork(path) => {
                println!("Commander loading & optimizing new network {:?}", path);
                let graph = load_graph(&path);

                // put it in an arc so we don't need to clone it a bunch of times
                let fused = Arc::new(graph);

                println!("Sending new network to executors");
                for sender in &graph_senders {
                    sender.send(Some(Arc::clone(&fused))).unwrap();
                }
            }
            Command::WaitForNewNetwork => {
                println!("Waiting for new network");
                for sender in &graph_senders {
                    sender.send(None).unwrap();
                }
            }
            Command::Stop => {
                //TODO this is probably not enough any more, we need to stop the gpu executors, cpu threads and rebatchers as well
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
