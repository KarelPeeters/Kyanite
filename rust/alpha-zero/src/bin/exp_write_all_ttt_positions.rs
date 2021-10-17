use board_game::ai::solver::solve_all_moves;
use board_game::board::{Board, BoardAvailableMoves};
use board_game::games::ttt::TTTBoard;
use board_game::util::game_stats::all_possible_boards;
use board_game::wdl::WDL;
use internal_iterator::InternalIterator;

use alpha_zero::mapping::binary_output::BinaryOutput;
use alpha_zero::mapping::ttt::TTTStdMapper;
use alpha_zero::network::ZeroEvaluation;
use alpha_zero::selfplay::simulation::{Position, Simulation};

fn main() -> std::io::Result<()> {
    let mut output = BinaryOutput::new("all_ttt", "ttt", TTTStdMapper)?;

    let positions = all_possible_boards(&TTTBoard::default(), false);

    for board in positions {
        let eval = solve_all_moves(&board, 20);
        let outcome = eval.value.to_outcome_wdl().unwrap().un_pov(board.next_player());

        let best_moves = eval.best_move.unwrap();
        let policy = board.available_moves().map(|mv| {
            if best_moves.contains(&mv) {
                1.0 / best_moves.len() as f32
            } else {
                0.0
            }
        }).collect();

        let zero_evaluation = ZeroEvaluation {
            wdl: WDL::nan(),
            policy,
        };
        let net_evaluation = ZeroEvaluation {
            wdl: WDL::nan(),
            policy: vec![f32::NAN; board.available_moves().count()],
        };

        let position = Position {
            board,
            should_store: true,
            zero_visits: 0,
            zero_evaluation,
            net_evaluation,
        };

        let simulation = Simulation {
            outcome,
            positions: vec![position],
        };

        output.append(simulation)?;
    }

    output.finish()?;
    Ok(())
}