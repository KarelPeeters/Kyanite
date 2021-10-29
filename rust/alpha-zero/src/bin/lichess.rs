use std::time::{Duration, Instant};

use board_game::board::Board;
use board_game::games::chess::ChessBoard;
use tokio_stream::StreamExt;
use unwrap_match::unwrap_match;

use alpha_zero::mapping::chess::ChessStdMapper;
use alpha_zero::network::cudnn::CudnnNetwork;
use alpha_zero::zero::wrapper::ZeroSettings;
use cuda_nn_eval::Device;
use licoricedev::client::{Lichess, LichessResult};
use licoricedev::models::board::{BoardState, GameFull};
use nn_graph::onnx::load_graph_from_onnx_path;

fn main() -> LichessResult<()> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { main_impl().await })
}

async fn main_impl() -> LichessResult<()> {
    let max_visits = 100_000;
    let max_fraction_time_used = 1.0 / 30.0;

    let path = std::fs::read_to_string("ignored/network_path.txt").unwrap();
    let graph = load_graph_from_onnx_path(path);
    let settings = ZeroSettings::new(100, 4.0);
    let mut network = CudnnNetwork::new(ChessStdMapper, graph, settings.batch_size, Device::new(0));

    let token = std::fs::read_to_string("ignored/lichess_token.txt")?;
    let lichess = Lichess::new(token);

    loop {
        // the games are already sorted by urgency by the lichess API
        let games = lichess.get_ongoing_games(50).await?;

        for game in games {
            if !game.is_my_turn {
                continue;
            }

            let mut state = lichess.stream_bot_game_state(&game.game_id).await?;
            if let Some(state) = state.next().await {
                println!("{:?}", state);
                let state: GameFull = unwrap_match!(state?, BoardState::GameFull(state) => state);
                let board = board_from_moves(&state.state.moves);
                println!("{}", board);

                let start = Instant::now();
                let tree = settings.build_tree(&board, &mut network, |tree| {
                    println!("{}, {:.2?}", tree.root_visits(), tree.wdl());
                    let fraction_time_used = (Instant::now() - start).as_secs_f32() / game.seconds_left as f32;
                    let visits = tree.root_visits();
                    visits > 0 && (visits >= max_visits || fraction_time_used >= max_fraction_time_used)
                });

                println!("{}", tree.display(1, true));
                let mv = tree.best_move();

                if let Err(e) = lichess.make_a_bot_move(&game.game_id, &mv.to_string(), false).await {
                    // can happen when the other player resigns or aborts the game
                    println!("Error while playing move: {:?}", e);
                }

                let message = format!(
                    "visits: {}, zero {:.2?} net {:.2?}",
                    tree.root_visits(), tree.wdl().to_slice(), tree[0].net_wdl.unwrap().to_slice(),
                );
                lichess.write_in_bot_chat(&game.game_id, "player", &message).await?;
            }
        }

        // wait for a bit
        std::thread::sleep(Duration::from_secs_f32(1.0));
    }
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