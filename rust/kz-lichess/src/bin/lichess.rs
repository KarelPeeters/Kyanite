use std::collections::{HashSet, VecDeque};
use std::time::{Duration, Instant};

use board_game::board::Board;
use board_game::games::chess::{ChessBoard, Rules};
use board_game::util::pathfind::pathfind_exact_length;
use itertools::Itertools;
use tokio_stream::StreamExt;

use cuda_nn_eval::Device;
use kz_core::mapping::chess::ChessStdMapper;
use kz_core::network::cudnn::CudaNetwork;
use kz_core::network::Network;
use kz_core::oracle::DummyOracle;
use kz_core::zero::node::UctWeights;
use kz_core::zero::step::FpuMode;
use kz_core::zero::tree::Tree;
use kz_core::zero::wrapper::ZeroSettings;
use licorice::client::{Lichess, LichessResult};
use licorice::models::board::{BoardState, GameFull};
use licorice::models::game::UserGame;
use nn_graph::onnx::load_graph_from_onnx_path;
use nn_graph::optimizer::{optimize_graph, OptimizerSettings};

const MAX_VISITS: u64 = 10_000_000;
const MAX_TIME: f32 = 60.0;
const MAX_TIME_FRACTION: f32 = 1.2 / 30.0;
const MAX_CACHE_SIZE: usize = 10;

type Cache = VecDeque<Tree<ChessBoard>>;

fn main() {
    let batch_size = (MAX_VISITS / 10).clamp(1, 128) as usize;

    let settings = ZeroSettings::new(batch_size, UctWeights::default(), false, FpuMode::Parent);
    println!("Using {:?}", settings);

    println!("Loading graph & constructing network");
    let path = std::fs::read_to_string("ignored/network_path.txt").unwrap();
    let graph = optimize_graph(&load_graph_from_onnx_path(path), OptimizerSettings::default());
    let mut network = CudaNetwork::new(ChessStdMapper, &graph, settings.batch_size, Device::new(0));

    let mut cache = Cache::default();

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { main_async(settings, &mut network, &mut cache).await })
}

async fn main_async(settings: ZeroSettings, network: &mut impl Network<ChessBoard>, cache: &mut Cache) {
    loop {
        if let Err(e) = main_inner(settings, network, cache).await {
            println!("Got error {:?}", e);
        }

        std::thread::sleep(Duration::from_secs(5));
    }
}

async fn main_inner(
    settings: ZeroSettings,
    network: &mut impl Network<ChessBoard>,
    cache: &mut Cache,
) -> LichessResult<()> {
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
                        make_move(&lichess, &game, &state, print, settings, network, cache).await?;
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

fn pop_cache_match(board: &ChessBoard, cache: &mut Cache) -> Option<Tree<ChessBoard>> {
    for (i, old_tree) in cache.iter().enumerate() {
        if let Some(moves) = pathfind_exact_length(old_tree.root_board(), board, 2) {
            if let Ok(new_tree) = old_tree.keep_moves(&moves) {
                cache.remove(i);
                return Some(new_tree);
            }
        }
    }

    None
}

async fn make_move(
    lichess: &Lichess,
    game: &UserGame,
    state: &GameFull,
    info: bool,
    settings: ZeroSettings,
    network: &mut impl Network<ChessBoard>,
    cache: &mut Cache,
) -> LichessResult<()> {
    let board = board_from_state(state);
    println!("{}", board);

    let mut tree = match pop_cache_match(&board, cache) {
        Some(tree) => {
            println!("Reusing tree with {} nodes", tree.root_visits());
            tree
        }
        None => {
            println!("Starting new tree");
            Tree::new(board)
        }
    };

    let start = Instant::now();
    let start_visits = tree.root_visits();

    settings.expand_tree(&mut tree, network, &DummyOracle, |tree| {
        let time_used = (Instant::now() - start).as_secs_f32();
        let fraction_time_used = time_used / game.seconds_left as f32;
        let visits = tree.root_visits();
        visits > 0 && (visits >= MAX_VISITS || time_used >= MAX_TIME || fraction_time_used >= MAX_TIME_FRACTION)
    });

    let time_used = Instant::now() - start;
    println!("Took {:?}", (time_used));
    println!(
        "GPU throughput: {:.2} evals/s",
        (tree.root_visits() - start_visits) as f32 / time_used.as_secs_f32()
    );

    println!("{}", tree.display(3, true, 5, false));
    let mv = tree.best_move().unwrap();

    if let Err(e) = lichess.make_a_bot_move(&game.game_id, &mv.to_string(), false).await {
        // can happen when the other player resigns or aborts the game
        println!("Error while playing move: {:?}", e);
    }

    if info {
        let pv = tree.principal_variation(3).iter().skip(1).join(" ");

        let message = format!(
            "visits: {}, depth: {:?}, pv: {}",
            tree.root_visits(),
            tree.depth_range(0),
            pv,
        );
        println!("Sending {:?}", message);
        lichess.write_in_bot_chat(&game.game_id, "player", &message).await?;

        let message = format!(
            "zero: {:.2?}, net: {:.2?}",
            tree.values().wdl.to_slice(),
            tree[0].net_values.unwrap().wdl.to_slice(),
        );
        println!("Sending {:?}", message);
        lichess.write_in_bot_chat(&game.game_id, "player", &message).await?;
    }

    cache.push_back(tree);
    while cache.len() > MAX_CACHE_SIZE {
        cache.pop_front();
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
