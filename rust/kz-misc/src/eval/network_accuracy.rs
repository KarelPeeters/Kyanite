use board_game::board::Board;
use decorum::N32;
use internal_iterator::InternalIterator;
use itertools::Itertools;

use kz_core::network::{Network, ZeroEvaluation};
use kz_util::{kdl_divergence, zip_eq_exact};

#[derive(Debug)]
pub struct Challenge<B> {
    pub board: B,
    pub solution: ZeroEvaluation<'static>,
    pub is_optimal: Option<Vec<bool>>,
}

#[derive(Debug)]
pub struct ProbDistrStats {
    pub accuracy: f32,

    pub actual_top: f32,
    pub solution_top: f32,

    pub captured_solution: f32,
    pub captured_actual: f32,

    pub kdl_actual: f32,
    pub kdl_solution: f32,
}

pub fn network_accuracy<B: Board>(network: &mut impl Network<B>, challenges: &[Challenge<B>]) {
    for Challenge { board, solution, is_optimal } in challenges {
        println!("{}", board);
        println!("Number of available moves: {}", board.available_moves().count());

        let eval = network.evaluate(board);

        println!("WDL:");
        println!("  solution: {:?}", solution.values.wdl);
        println!("  actual: {:?}", eval.values.wdl);
        let wdl_stats = prob_distr_stats(&eval.values.wdl.to_slice(), &solution.values.wdl.to_slice());
        println!("  stats: {:?}", wdl_stats);

        println!("Policy:");
        println!("  solution: {:?}", solution.policy);
        println!("  actual: {:?}", eval.policy);
        let policy_stats = prob_distr_stats(&eval.policy, &solution.policy);
        println!("  stats: {:?}", policy_stats);

        if let Some(optimal_moves) = is_optimal {
            let optimal_p = zip_eq_exact(&*eval.policy, optimal_moves)
                .map(|(&p, &w)| if w { p } else { 0.0 })
                .sum::<f32>();
            println!("  optimal_p: {:?}", optimal_p);
        }
    }
}

fn prob_distr_stats(actual: &[f32], solution: &[f32]) -> ProbDistrStats {
    assert_eq!(actual.len(), solution.len());

    let actual_argmax = argmax(actual);
    let solution_argmax = argmax(solution);

    ProbDistrStats {
        accuracy: (actual_argmax == solution_argmax) as u8 as f32,
        actual_top: actual[actual_argmax],
        solution_top: solution[actual_argmax],
        captured_solution: solution[actual_argmax],
        captured_actual: actual[solution_argmax],
        kdl_actual: kdl_divergence(actual, solution),
        kdl_solution: kdl_divergence(solution, actual),
    }
}

fn argmax(data: &[f32]) -> usize {
    data.iter()
        .position_max_by_key(|&&f| N32::from_inner(f))
        .unwrap()
}