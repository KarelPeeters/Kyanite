use board_game::board::{Outcome, Player};
use board_game::pov::{NonPov, ScalarAbs};
use serde::{Deserialize, Serialize};

use crate::zero::range::IdxRange;
use crate::zero::values::ZeroValuesAbs;

// TODO look at the size of this struct and think about making it smaller
//   (but first try padding it so see if that makes it slower)
#[derive(Debug, Clone)]
pub struct Node<M> {
    // Potentially update Tree::keep_moves when this struct gets new fields.
    /// The parent node.
    pub parent: Option<usize>,
    /// The move that was just made to get to this node. Is `None` only for the root node.
    pub last_move: Option<M>,
    /// The children of this node. Is `None` if this node has not been visited yet.
    pub children: Option<IdxRange>,

    /// The number of non-virtual visits for this node and its children.
    pub complete_visits: u64,
    /// The number of virtual visits for this node and its children.
    pub virtual_visits: u64,
    /// The sum of final values found in children of this node. Should be divided by `visits` to get the expected value.
    /// Does not include virtual visits.
    pub sum_values: ZeroValuesAbs,

    /// The data returned by the network for this position.
    /// If `None` and the node has children this means this is a virtual node with data to be filled in later.
    pub net_values: Option<ZeroValuesAbs>,
    /// The policy/prior probability as evaluated by the network when the parent node was expanded.
    pub net_policy: f32,
}

#[derive(Debug, Copy, Clone)]
pub struct Uct {
    // value, range -1..1
    pub v: f32,
    // exploration term, range 0..inf
    pub u: f32,
    // moves left term, range -inf..inf
    pub m: f32,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct UctWeights {
    pub exploration_weight: f32,

    pub moves_left_weight: f32,
    pub moves_left_clip: f32,
    pub moves_left_sharpness: f32,
}

impl Default for UctWeights {
    fn default() -> Self {
        UctWeights {
            exploration_weight: 2.0,
            moves_left_weight: 0.03,
            moves_left_clip: 20.0,
            moves_left_sharpness: 0.5,
        }
    }
}

impl Uct {
    pub fn nan() -> Uct {
        Uct {
            v: f32::NAN,
            u: f32::NAN,
            m: f32::NAN,
        }
    }

    pub fn total(self, weights: UctWeights) -> f32 {
        let Uct { v, u, m } = self;

        let m_unit = if weights.moves_left_weight == 0.0 {
            0.0
        } else {
            assert!(m.is_finite(), "Invalid moves_left value {}", m);

            let m_clipped = m.clamp(-weights.moves_left_clip, weights.moves_left_clip);
            (weights.moves_left_sharpness * m_clipped * -v).clamp(-1.0, 1.0)
        };

        v + weights.exploration_weight * u + weights.moves_left_weight * m_unit
    }
}

#[derive(Debug)]
pub struct NotYetVisited;

impl<N> Node<N> {
    pub(super) fn new(parent: Option<usize>, last_move: Option<N>, p: f32) -> Self {
        Node {
            parent,
            last_move,
            children: None,

            complete_visits: 0,
            virtual_visits: 0,
            sum_values: ZeroValuesAbs::default(),

            net_values: None,
            net_policy: p,
        }
    }

    pub fn total_visits(&self) -> u64 {
        self.complete_visits + self.virtual_visits
    }

    /// The (normalized) values of this node.
    pub fn values(&self) -> ZeroValuesAbs {
        self.sum_values / self.complete_visits as f32
    }

    /// Get the outcome of this node if it's terminal.
    /// * `Err(NotYetVisited)` means we don't know yet because this node has not been visited yet,
    /// * `Ok(None)` means this node is not terminal.
    /// * `Ok(Some(outcome))` is the outcome of this node
    pub fn outcome(&self) -> Result<Option<Outcome>, NotYetVisited> {
        if self.children.is_none() {
            if self.total_visits() > 0 {
                assert_eq!(self.virtual_visits, 0, "Terminal node cannot have virtual visits");
                let outcome = self
                    .values()
                    .wdl_abs
                    .try_to_outcome()
                    .unwrap_or_else(|| panic!("Unexpected wdl in values {:?} for terminal node", self.values()));
                Ok(Some(outcome))
            } else {
                Err(NotYetVisited)
            }
        } else {
            Ok(None)
        }
    }

    pub fn uct(
        &self,
        parent_total_visits: u64,
        parent_moves_left: f32,
        fpu: ZeroValuesAbs,
        use_value: bool,
        pov: Player,
    ) -> Uct {
        if parent_total_visits == 0 {
            return Uct::nan();
        }
        let total_visits = self.total_visits();

        let fpu_value = select_value(fpu, use_value).pov(pov).value;

        let total_value = select_value(self.sum_values, use_value).pov(pov).value;
        let node_value = (total_value - self.virtual_visits as f32) / total_visits as f32;

        let (v, m) = if total_visits == 0 {
            // don't even bother with moves_left if we don't have any information
            (fpu_value, 0.0)
        } else {
            // this node has been visited, so we know parent_moves_left is also a useful value
            let m = self.values().moves_left - parent_moves_left - 1.0;
            (node_value, m)
        };

        let u = self.net_policy * ((parent_total_visits - 1) as f32).sqrt() / (1 + total_visits) as f32;

        Uct { v, u, m }
    }
}

fn select_value(values: ZeroValuesAbs, use_value: bool) -> ScalarAbs<f32> {
    if use_value {
        values.value_abs
    } else {
        values.wdl_abs.value()
    }
}
