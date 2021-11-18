use board_game::ai::solver::solve_all_moves;
use board_game::board::Board;
use internal_iterator::InternalIterator;
use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::mapping::binary_output::BinaryOutput;
use crate::mapping::BoardMapper;
use crate::network::ZeroEvaluation;
use crate::selfplay::simulation::{Position, Simulation};
use crate::util::PrintThroughput;
use crate::zero::node::ZeroValues;

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
                zero_visits: 0,
                net_evaluation: ZeroEvaluation {
                    values: ZeroValues::nan(),
                    policy: vec![f32::NAN; mv_count],
                },
                zero_evaluation: ZeroEvaluation {
                    values: ZeroValues::nan(),
                    policy,
                },
            });

            board.play(mv);
        };

        let simulation = Simulation { outcome, positions };
        bin.append(simulation)?;

        pt.update(1);
    }

    Ok(())
}