use std::collections::HashSet;
use std::time::{Duration, Instant};

use board_game::board::Board;
use board_game::games::chess::ChessBoard;
use itertools::Itertools;
use tokio_stream::StreamExt;

use alpha_zero::mapping::chess::ChessStdMapper;
use alpha_zero::network::cudnn::CudnnNetwork;
use alpha_zero::network::Network;
use alpha_zero::oracle::DummyOracle;
use alpha_zero::zero::step::FpuMode;
use alpha_zero::zero::wrapper::ZeroSettings;
use cuda_nn_eval::Device;
use licoricedev::client::{Lichess, LichessResult};
use licoricedev::models::board::{BoardState, GameFull};
use licoricedev::models::game::UserGame;
use nn_graph::onnx::load_graph_from_onnx_path;

const MAX_VISITS: u64 = 100_000;
const MAX_FRACTION_TIME_USED: f32 = 1.0 / 30.0;

fn main() {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { main_async().await })
}

async fn main_async() {
    loop {
        if let Err(e) = main_inner().await {
            println!("Got error {:?}", e);
        }

        std::thread::sleep(Duration::from_secs(10));
    }
}

async fn main_inner() -> LichessResult<()> {
    let path = std::fs::read_to_string("ignored/network_path.txt").unwrap();
    let graph = load_graph_from_onnx_path(path);
    let settings = ZeroSettings::new(128, 4.0, false, FpuMode::Parent);
    let mut network = CudnnNetwork::new(ChessStdMapper, graph, settings.batch_size, Device::new(0));

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
                        make_move(&lichess, &game, &state, print, settings, &mut network).await?;
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
    let board = board_from_moves(&state.state.moves);
    println!("{}", board);

    let start = Instant::now();
    let tree = settings.build_tree(&board, network, &DummyOracle, |tree| {
        let fraction_time_used = (Instant::now() - start).as_secs_f32() / game.seconds_left as f32;
        let visits = tree.root_visits();
        visits > 0 && (visits >= MAX_VISITS || fraction_time_used >= MAX_FRACTION_TIME_USED)
    });

    println!("{}", tree.display(3, true, 5));
    let mv = tree.best_move();

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

fn board_from_moves(moves: &str) -> ChessBoard {
    let mut board = ChessBoard::default();

    if !moves.is_empty() {
        for mv in moves.split(' ') {
            let mv = board.parse_move(mv)
                .unwrap_or_else(|e| panic!("Failed to parse move '{}' with error {:?}", mv, e));
            board.play(mv)
        }
    }

    board
}