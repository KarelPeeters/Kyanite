use std::borrow::Cow;

use board_game::ai::solver::solve_all_moves;
use board_game::board::Board;
use internal_iterator::InternalIterator;
use rand::seq::SliceRandom;
use rand::thread_rng;

use kz_core::mapping::BoardMapper;
use kz_core::network::ZeroEvaluation;
use kz_core::zero::node::ZeroValues;
use kz_selfplay::binary_output::BinaryOutput;
use kz_selfplay::simulation::{Position, Simulation};
use kz_util::PrintThroughput;

pub fn append_random_games_to_bin<B: Board, M: BoardMapper<B>>(start: &B, count: usize, solver_depth: u32, bin: &mut BinaryOutput<B, M>) -> std::io::Result<()> {
    let mut rng = thread_rng();
    let mut pt = PrintThroughput::new("games");

    for _ in 0..count {
        let mut board = start.clone();
        let mut positions = vec![];

        let outcome = loop {
            if let Some(outcome) = board.outcome() {
                break outcome;
            }

            let mv_count = board.available_moves().count();

            let (mv, policy) = if solver_depth != 0 {
                // pick a random best move and use all best moves for the policy
                let solution = solve_all_moves(&board, solver_depth);
                let best_moves = solution.best_move.unwrap();

                let policy = board.available_moves()
                    .map(|mv: B::Move| best_moves.contains(&mv) as u8 as f32 / best_moves.len() as f32)
                    .collect();
                let mv = *best_moves.choose(&mut rng).unwrap();

                (mv, policy)
            } else {
                // pick a random best move with uniform policy
                let mv = board.random_available_move(&mut rng);
                let policy = vec![1.0 / mv_count as f32; mv_count];

                (mv, policy)
            };

            positions.push(Position {
                board: board.clone(),
                should_store: true,
                played_mv: mv,
                zero_visits: 0,
                net_evaluation: ZeroEvaluation {
                    values: ZeroValues::nan(),
                    policy: Cow::Owned(vec![f32::NAN; mv_count]),
                },
                zero_evaluation: ZeroEvaluation {
                    values: ZeroValues::nan(),
                    policy: Cow::Owned(policy),
                },
            });

            board.play(mv);
        };

        bin.append(&Simulation { outcome, positions })?;
        pt.update_delta(1);
    }

    Ok(())
}