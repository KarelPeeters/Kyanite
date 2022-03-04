use decorum::N32;
use itertools::Itertools;
use rand::distributions::WeightedIndex;
use rand::Rng;

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
        MoveSelector {
            temperature,
            zero_temp_move_count,
        }
    }

    /// Always select the move with the maximum policy, ie. temperature 0.
    pub fn zero_temp() -> Self {
        Self::new(0.0, 0)
    }

    pub fn constant_temp(temperature: f32) -> Self {
        MoveSelector {
            temperature,
            zero_temp_move_count: u32::MAX,
        }
    }
}

impl MoveSelector {
    pub fn select(&self, move_count: u32, policy: &[f32], rng: &mut impl Rng) -> usize {
        let temperature = if move_count >= self.zero_temp_move_count {
            0.0
        } else {
            self.temperature
        };
        assert!(temperature >= 0.0);

        // we handle the extreme cases separately, in theory that would not be necessary but they're degenerate
        if temperature == 0.0 {
            // pick the best move
            policy.iter().copied().map(N32::from).position_max().unwrap()
        } else if temperature == f32::INFINITY {
            // pick a random move
            rng.gen_range(0..policy.len())
        } else {
            // pick according to `policy ** (1/temperature)`
            let policy_temp = policy.iter().map(|p| p.powf(1.0 / temperature));
            let distr = WeightedIndex::new(policy_temp).unwrap();
            rng.sample(distr)
        }
    }
}
