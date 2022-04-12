use board_game::ai::solver::solve_all_moves;
use board_game::board::{Board, Outcome};
use board_game::games::ataxx::AtaxxBoard;
use board_game::games::chess::ChessBoard;
use board_game::games::sttt::STTTBoard;
use board_game::games::ttt::TTTBoard;
use board_game::wdl::OutcomeWDL;
use clap::Parser;
use kz_core::mapping::ataxx::AtaxxStdMapper;
use kz_core::mapping::chess::ChessStdMapper;
use kz_core::mapping::sttt::STTTStdMapper;
use kz_core::mapping::ttt::TTTStdMapper;
use kz_core::mapping::BoardMapper;
use kz_selfplay::binary_output::BinaryOutput;
use kz_selfplay::server::protocol::Game;
use std::borrow::Cow;
use std::path::PathBuf;

use internal_iterator::InternalIterator;
use kz_core::network::dummy::{uniform_policy, uniform_values};
use kz_core::network::ZeroEvaluation;
use kz_core::zero::node::ZeroValues;
use kz_selfplay::simulation::{Position, Simulation};
use kz_util::PrintThroughput;
use rand::seq::SliceRandom;
use rand::thread_rng;

#[derive(Debug, Parser)]
struct Args {
    #[clap(long)]
    game: String,
    #[clap(long)]
    max_game_length: u64,
    #[clap(long)]
    game_count: u64,
    #[clap(long)]
    solver_depth: u32,

    bin_path: PathBuf,
}

fn main() {
    let args = Args::parse();

    let game = Game::parse(&args.game).unwrap();

    match game {
        Game::TTT => main_impl(TTTBoard::default, TTTStdMapper, args),
        Game::STTT => main_impl(STTTBoard::default, STTTStdMapper, args),
        Game::Chess => main_impl(ChessBoard::default, ChessStdMapper, args),
        Game::Ataxx { size } => main_impl(|| AtaxxBoard::diagonal(size), AtaxxStdMapper::new(size), args),
    }
}

fn main_impl<B: Board, M: BoardMapper<B>>(mut start_pos: impl FnMut() -> B, mapper: M, args: Args) {
    let Args {
        game,
        max_game_length,
        game_count,
        bin_path,
        solver_depth,
    } = args;

    let mut rng = thread_rng();
    let mut output = BinaryOutput::new(bin_path, &game, mapper).unwrap();
    let mut tp = PrintThroughput::new("games");

    for _ in 0..game_count {
        let mut board = start_pos();
        let mut positions = vec![];

        for _ in 0..max_game_length {
            if board.is_done() {
                break;
            }

            let net_eval = ZeroEvaluation {
                values: uniform_values(),
                policy: Cow::Owned(uniform_policy(board.available_moves().count())),
            };

            let solution = solve_all_moves(&board, solver_depth);
            let (zero_eval, mv) = if let Some(moves) = solution.best_move {
                let outcome = solution.value.to_outcome_wdl().unwrap_or(OutcomeWDL::Draw);
                let policy = board
                    .available_moves()
                    .map(|mv: B::Move| moves.contains(&mv) as u8 as f32 / moves.len() as f32)
                    .collect();

                let zero_eval = ZeroEvaluation {
                    values: ZeroValues::from_outcome(outcome, 0.0),
                    policy: Cow::Owned(policy),
                };

                let mv = *moves.choose(&mut rng).unwrap();

                (zero_eval, mv)
            } else {
                (net_eval.clone(), board.random_available_move(&mut rng))
            };

            let position = Position {
                board: board.clone(),
                should_store: true,
                played_mv: mv,
                zero_visits: 0,
                zero_evaluation: zero_eval,
                net_evaluation: net_eval,
            };
            positions.push(position);

            board.play(mv);
        }

        let outcome = board.outcome().unwrap_or(Outcome::Draw);
        let sim = Simulation { outcome, positions };

        output.append(&sim).unwrap();
        tp.update_delta(1);
    }

    output.finish().unwrap();
}
