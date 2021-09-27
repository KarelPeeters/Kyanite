use decorum::N32;
use itertools::Itertools;
use rand::distributions::WeightedIndex;
use rand::Rng;

use board_game::board::{Board, Outcome};
use board_game::wdl::{POV, WDL};
use crate::network::ZeroEvaluation;

pub trait Output<B> {
    fn append(&mut self, simulation: Simulation<B>);
}

#[derive(Debug, Copy, Clone)]
pub struct MoveSelector {
    /// The temperature applied to the policy before sampling. Can be any positive value.
    /// * `0.0`: always pick the move with highest policy
    /// * `inf`: pick a completely random move
    pub temperature: f32,

    /// After this number of moves, use temperature zero to always select the best move.
    pub zero_temp_move_count: u32,
}

impl MoveSelector {
    pub fn new(temperature: f32, zero_temp_move_count: u32) -> Self {
        MoveSelector { temperature, zero_temp_move_count }
    }

    /// Always select the move with the maximum policy, ie. temperature 0.
    pub fn zero_temp() -> Self {
        Self::new(0.0, 0)
    }

    pub fn constant_temp(temperature: f32) -> Self {
        MoveSelector { temperature, zero_temp_move_count: u32::MAX }
    }
}

/// A full game.
#[derive(Debug)]
pub struct Simulation<B> {
    pub outcome: Outcome,
    pub positions: Vec<Position<B>>,
}

/// A single position in a game.
#[derive(Debug)]
pub struct Position<B> {
    pub board: B,
    pub should_store: bool,

    /// The enhanced MCTS evaluation, not the immediate network output.
    pub evaluation: ZeroEvaluation,
}

/// A position with the final `WDL`, zero `WDL` and zero policy.
/// Returned by iterating over a `Simulation`.
#[derive(Debug)]
pub struct StorePosition<B: Board> {
    pub board: B,
    pub final_wdl: WDL<f32>,
    pub evaluation: ZeroEvaluation,
}

impl<B: Board + 'static> Simulation<B> {
    pub fn iter(self) -> impl Iterator<Item=StorePosition<B>> {
        let Simulation { outcome, positions } = self;

        positions.into_iter()
            .filter(|pos| pos.should_store)
            .map(move |pos| {
                let final_wdl = outcome.pov(pos.board.next_player()).to_wdl();
                StorePosition { board: pos.board, final_wdl, evaluation: pos.evaluation }
            })
    }
}

impl MoveSelector {
    pub fn select(&self, move_count: u32, policy: impl IntoIterator<Item=f32>, rng: &mut impl Rng) -> usize {
        let temperature = if move_count >= self.zero_temp_move_count { 0.0 } else { self.temperature };
        assert!(temperature >= 0.0);

        let policy = policy.into_iter();

        // we handle the extreme cases separately, in theory that would not be necessary but they're degenerate
        if temperature == 0.0 {
            // pick the best move
            policy.map(N32::from).position_max().unwrap()
        } else if temperature == f32::INFINITY {
            // pick a random move
            rng.gen_range(0..policy.count())
        } else {
            // pick according to `policy ** (1/temperature)`
            let policy_temp = policy.map(|p| p.powf(1.0 / temperature));
            let distr = WeightedIndex::new(policy_temp).unwrap();
            rng.sample(distr)
        }
    }
}
