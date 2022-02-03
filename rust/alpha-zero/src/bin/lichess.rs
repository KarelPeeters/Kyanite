use std::collections::HashSet;
use std::time::{Duration, Instant};

use board_game::board::Board;
use board_game::games::chess::{ChessBoard, Rules};
use itertools::Itertools;
use tokio_stream::StreamExt;

use alpha_zero::mapping::chess::ChessStdMapper;
use alpha_zero::network::cudnn::CudnnNetwork;
use alpha_zero::network::Network;
use alpha_zero::oracle::DummyOracle;
use alpha_zero::zero::node::UctWeights;
use alpha_zero::zero::step::FpuMode;
use alpha_zero::zero::wrapper::ZeroSettings;
use cuda_nn_eval::Device;
use licoricedev::client::{Lichess, LichessResult};
use licoricedev::models::board::{BoardState, GameFull};
use licoricedev::models::game::UserGame;
use nn_graph::onnx::load_graph_from_onnx_path;
use nn_graph::optimizer::{optimize_graph, OptimizerSettings};

const MAX_VISITS: u64 = 10_000_000;
const MAX_TIME: f32 = 60.0;
const MAX_TIME_FRACTION: f32 = 1.2 / 30.0;

fn main() {
    // TODO why this high exploration weight?
    let settings = ZeroSettings::new(128, UctWeights::default(), false, FpuMode::Parent);
    println!("Using {:?}", settings);

    println!("Loading graph & constructing network");
    let path = std::fs::read_to_string("ignored/network_path.txt").unwrap();
    let graph = optimize_graph(&load_graph_from_onnx_path(path), OptimizerSettings::default());
    let mut network = CudnnNetwork::new(ChessStdMapper, graph, settings.batch_size, Device::new(0));

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { main_async(settings, &mut network).await })
}

async fn main_async(settings: ZeroSettings, network: &mut impl Network<ChessBoard>) {
    loop {
        if let Err(e) = main_inner(settings, network).await {
            println!("Got error {:?}", e);
        }

        std::thread::sleep(Duration::from_secs(10));
    }
}

async fn main_inner(settings: ZeroSettings, network: &mut impl Network<ChessBoard>) -> LichessResult<()> {
    println!("Connecting to lichess");
    let token = std::fs::read_to_string("ignored/lichess_token.txt")?;
    let lichess = Lichess::new(token);

    let mut info_game_ids: HashSet<String> = Default::default();

    loop {
        let mut was_my_turn = false;

        // the games are already sorted by urgency by the lichess API
        let games = lichess.get_ongoing_games(50).await?;

        for game in games {
            if !game.is_my_turn {
                continue;
            }
            was_my_turn = true;

            let mut state_stream = lichess.stream_bot_game_state(&game.game_id).await?;
            if let Some(state) = state_stream.next().await.transpose()? {
                println!("{:?}", state);

                match state {
                    BoardState::GameState(state) => {
                        println!("Received partial state {:?}", state);
                    }
                    BoardState::ChatLine(line) => {
                        println!("Received {:?}", line);

                        match &*line.text {
                            "info start" => {
                                info_game_ids.insert(game.game_id.clone());
                            }
                            "info stop" => {
                                info_game_ids.remove(&game.game_id);
                            }
                            _ => {}
                        }
                    }
                    BoardState::GameFull(state) => {
                        let print = info_game_ids.contains(&state.id);
                        make_move(&lichess, &game, &state, print, settings, network).await?;
                    }
                }
            }
        }

        if !was_my_turn {
            // wait for a bit
            std::thread::sleep(Duration::from_secs(1));
        }
    }
}

async fn make_move(
    lichess: &Lichess,
    game: &UserGame,
    state: &GameFull,
    info: bool,
    settings: ZeroSettings,
    network: &mut impl Network<ChessBoard>,
) -> LichessResult<()> {
    let board = board_from_state(state);
    println!("{}", board);

    let start = Instant::now();
    let tree = settings.build_tree(&board, network, &DummyOracle, |tree| {
        let time_used = (Instant::now() - start).as_secs_f32();
        let fraction_time_used = time_used / game.seconds_left as f32;
        let visits = tree.root_visits();
        visits > 0 && (visits >= MAX_VISITS || time_used >= MAX_TIME || fraction_time_used >= MAX_TIME_FRACTION)
    });

    let time_used = Instant::now() - start;
    println!("Took {:?}", (time_used));
    println!("GPU throughput: {:.2} evals.s", tree.root_visits() as f32 / time_used.as_secs_f32());

    println!("{}", tree.display(3, true, 5));
    let mv = tree.best_move().unwrap();

    if let Err(e) = lichess.make_a_bot_move(&game.game_id, &mv.to_string(), false).await {
        // can happen when the other player resigns or aborts the game
        println!("Error while playing move: {:?}", e);
    }

    if info {
        let pv = tree.principal_variation(3).iter().skip(1).join(" ");

        let message = format!(
            "visits: {}, depth: {:?}, pv: {}",
            tree.root_visits(), tree.depth_range(0), pv,
        );
        println!("Sending {:?}", message);
        lichess.write_in_bot_chat(&game.game_id, "player", &message).await?;

        let message = format!(
            "zero: {:.2?}, net: {:.2?}",
            tree.values().wdl.to_slice(), tree[0].net_values.unwrap().wdl.to_slice(),
        );
        println!("Sending {:?}", message);
        lichess.write_in_bot_chat(&game.game_id, "player", &message).await?;
    }

    Ok(())
}

fn board_from_state(game: &GameFull) -> ChessBoard {
    let mut board = if game.initial_fen == "startpos" {
        ChessBoard::default()
    } else {
        ChessBoard::new_without_history_fen(&game.initial_fen, Rules::default())
    };

    if !game.state.moves.is_empty() {
        for mv in game.state.moves.split(' ') {
            let mv = board.parse_move(mv).unwrap();
            board.play(mv)
        }
    }

    board
}