use std::fs::File;
use std::io::{BufRead, BufReader};
use std::io::Write;
use std::time::Instant;

use board_game::board::{Board, BoardMoves};
use board_game::games::chess::{ChessBoard, Rules};
use board_game::wdl::WDL;
use crossbeam::channel::{Receiver, RecvError, Sender, TryRecvError};
use internal_iterator::InternalIterator;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use vampirc_uci::UciMessage;

use alpha_zero::mapping::chess::ChessStdMapper;
use alpha_zero::network::cudnn::CudnnNetwork;
use alpha_zero::oracle::DummyOracle;
use alpha_zero::zero::step::FpuMode;
use alpha_zero::zero::tree::Tree;
use alpha_zero::zero::wrapper::ZeroSettings;
use cuda_nn_eval::Device;
use nn_graph::onnx::load_graph_from_onnx_path;

const INFO_PERIOD: f32 = 0.5;

fn main() -> std::io::Result<()> {
    // io
    let (sender, receiver) = crossbeam::channel::unbounded();
    std::thread::spawn(|| io_thread(sender).unwrap());

    let mut debug = File::create("kzero_log.txt")?;
    let log = &mut debug;

    // search settings
    let path = "C:/Documents/Programming/STTT/AlphaZero/data/supervised/lichess_huge/network_5140.onnx";
    let batch_size = 100;
    let settings = ZeroSettings::new(batch_size, 2.0, false, FpuMode::Parent);

    let graph = load_graph_from_onnx_path(path);
    let mut network = CudnnNetwork::new(ChessStdMapper, graph, batch_size, Device::new(0));

    // state
    let mut tree = None;
    let mut searching = false;

    loop {
        // search until we receive a message
        if searching {
            if let Some(tree) = &mut tree {
                let mut prev_send = Instant::now();

                settings.expand_tree(tree, &mut network, &DummyOracle, |tree| {
                    let now = Instant::now();
                    if tree.root_visits() > 0 && (now - prev_send).as_secs_f32() > INFO_PERIOD {
                        let root = &tree[0];

                        let mut children: Vec<_> = root.children.unwrap().iter().collect();
                        children.sort_by_key(|&c| tree[c].complete_visits);
                        children.reverse();

                        for (i, &child_index) in children.iter().enumerate() {
                            let child = &tree[child_index];
                            let wdl: WDL<u32> = (child.values().wdl * 1000.0).cast();
                            let (min_depth, max_depth) = tree.depth_range(child_index);

                            println!(
                                "info depth {} seldepth {} nodes {} wdl {} {} {} multipv {} pv {}",
                                min_depth, max_depth, tree.root_visits(),
                                wdl.win, wdl.draw, wdl.loss,
                                i + 1, child.last_move.unwrap(),
                            )
                        }

                        // writeln!(log, "{}", tree.display(1, true)).unwrap();
                        prev_send = now;
                    }

                    !receiver.is_empty()
                });
            }
        }

        // process all messages
        while let Some(message) = receive(&receiver, !searching).unwrap() {
            writeln!(log, "> {}", message)?;

            match message {
                UciMessage::Uci => {
                    println!("id kZero");
                    println!("uciok");
                }
                UciMessage::IsReady => {
                    println!("readyok")
                }
                UciMessage::Position { startpos, fen, moves } => {
                    let mut board = match (startpos, fen) {
                        (true, None) => ChessBoard::default(),
                        (false, Some(fen)) => {
                            ChessBoard::new_without_history_fen(fen.as_str(), Rules::default())
                        }
                        _ => panic!("Invalid position command")
                    };

                    for mv in moves {
                        board.play(mv);
                    }

                    writeln!(log, "setting curr_board to {}", board)?;
                    tree = Some(Tree::new(board));
                }
                UciMessage::Go { .. } => {
                    searching = true;
                }
                UciMessage::Stop => {
                    searching = false;

                    if let Some(tree) = &tree {
                        let best_move = if tree.root_visits() > 0 {
                            tree.best_move()
                        } else {
                            let moves: Vec<_> = tree.root_board().available_moves().collect();
                            *moves.choose(&mut thread_rng()).unwrap()
                        };

                        println!("bestmove {}", best_move);
                    }
                }
                UciMessage::UciNewGame => {
                    tree = None;
                }
                UciMessage::Quit => {
                    return Ok(());
                }
                UciMessage::Debug(_) => {}
                UciMessage::Register { .. } => {}
                UciMessage::SetOption { .. } => {}
                UciMessage::PonderHit => {}
                UciMessage::Id { .. } => {}
                UciMessage::UciOk => {}
                UciMessage::ReadyOk => {}
                UciMessage::BestMove { .. } => {}
                UciMessage::CopyProtection(_) => {}
                UciMessage::Registration(_) => {}
                UciMessage::Option(_) => {}
                UciMessage::Info(_) => {}
                UciMessage::Unknown(_, _) => {}
            }
        }
    }
}

fn receive<T>(receiver: &Receiver<T>, blocking: bool) -> Result<Option<T>, RecvError> {
    if blocking {
        receiver.recv().map(Some)
    } else {
        match receiver.try_recv() {
            Ok(value) => Ok(Some(value)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(RecvError),
        }
    }
}

fn io_thread(sender: Sender<UciMessage>) -> std::io::Result<()> {
    // io
    let stdin = std::io::stdin();
    let mut stdin = BufReader::new(stdin.lock());

    // message loop
    let mut line = String::new();

    loop {
        stdin.read_line(&mut line)?;
        let message = vampirc_uci::parse_one(&line);
        sender.send(message).unwrap();
        line.clear();
    }
}
