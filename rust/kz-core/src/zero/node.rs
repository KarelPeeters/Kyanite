use std::fmt::{Display, Formatter};

use board_game::wdl::{Flip, OutcomeWDL, WDL};
use serde::{Deserialize, Serialize};

use crate::zero::range::IdxRange;

/// The data that is accumulated in a node.
#[derive(Debug, Copy, Clone, Default)]
pub struct ZeroValues {
    pub value: f32,
    pub wdl: WDL<f32>,
    pub moves_left: f32,
}

#[derive(Debug, Clone)]
// TODO look at the size of this struct and think about making it smaller
//   (but first try padding it so see if that makes it slower)
pub struct Node<M> {
    // Potentially update Tree::keep_moves when this struct gets new fields.<
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
    pub sum_values: ZeroValues,

    /// The data returned by the network for this position.
    /// If `None` and the node has children this means this is a virtual node with data to be filled in later.
    pub net_values: Option<ZeroValues>,
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
            sum_values: ZeroValues::default(),

            net_values: None,
            net_policy: p,
        }
    }

    pub fn total_visits(&self) -> u64 {
        self.complete_visits + self.virtual_visits
    }

    /// The (normalized) data of this node from the POV of the player that could play this move.
    /// Does not include virtual visits.
    pub fn values(&self) -> ZeroValues {
        self.sum_values / self.complete_visits as f32
    }

    /// The same as [Self::value] except that virtual visits are included.
    pub fn total_data(&self) -> ZeroValues {
        self.sum_values.add_virtual(self.virtual_visits) / self.total_visits() as f32
    }

    /// Get the outcome of this node if it's terminal.
    /// * `Err(NotYetVisited)` means we don't know yet because this node has not been visited yet,
    /// * `Ok(None)` means this node is not terminal.
    /// * `Ok(Some(outcome))` is the outcome of this node
    pub fn outcome(&self) -> Result<Option<OutcomeWDL>, NotYetVisited> {
        if self.children.is_none() {
            if self.total_visits() > 0 {
                let outcome = self
                    .total_data()
                    .wdl
                    .try_to_outcome_wdl()
                    .unwrap_or_else(|| panic!("Unexpected wdl {:?} for terminal node", self.total_data().wdl));
                Ok(Some(outcome))
            } else {
                Err(NotYetVisited)
            }
        } else {
            Ok(None)
        }
    }

    pub fn uct(&self, parent_total_visits: u64, fpu: ZeroValues, use_value: bool) -> Uct {
        if parent_total_visits == 0 {
            return Uct::nan();
        }

        let total_visits = self.total_visits();

        let data = if total_visits == 0 { fpu } else { self.total_data() };

        let v = if use_value { data.value } else { data.wdl.value() };

        let u = self.net_policy * ((parent_total_visits - 1) as f32).sqrt() / (1 + total_visits) as f32;
        //TODO make sure to remove this -1 if we ever split ZeroValues.flip() into child() and parent()
        let m = data.moves_left - (fpu.moves_left - 1.0);

        Uct { v, u, m }
    }
}

impl ZeroValues {
    pub fn from_outcome(outcome: OutcomeWDL, moves_left: f32) -> Self {
        ZeroValues {
            value: outcome.sign(),
            wdl: outcome.to_wdl(),
            moves_left,
        }
    }

    pub fn nan() -> Self {
        ZeroValues {
            value: f32::NAN,
            wdl: WDL::nan(),
            moves_left: f32::NAN,
        }
    }

    /// The value that should be accumulated in the parent node of this value.
    pub fn parent(&self) -> Self {
        ZeroValues {
            value: -self.value,
            wdl: self.wdl.flip(),
            moves_left: self.moves_left + 1.0,
        }
    }

    pub fn add_virtual(&self, virtual_visits: u64) -> Self {
        let virtual_visits = virtual_visits as f32;
        ZeroValues {
            value: self.value - virtual_visits,
            wdl: self.wdl + WDL::new(0.0, 0.0, virtual_visits),
            //TODO make sure this behaves correctly in the full impl
            moves_left: self.moves_left,
        }
    }
}

impl std::ops::Add<Self> for ZeroValues {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        ZeroValues {
            value: self.value + rhs.value,
            wdl: self.wdl + rhs.wdl,
            moves_left: self.moves_left + rhs.moves_left,
        }
    }
}

impl std::ops::AddAssign<Self> for ZeroValues {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl std::ops::Div<f32> for ZeroValues {
    type Output = ZeroValues;

    fn div(self, rhs: f32) -> Self::Output {
        ZeroValues {
            value: self.value / rhs,
            wdl: self.wdl / rhs,
            moves_left: self.moves_left / rhs,
        }
    }
}

impl ZeroValues {
    pub const FORMAT_SUMMARY: &'static str = "v w/d/l ml";
}

impl Display for ZeroValues {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.3}, {:.3}/{:.3}/{:.3}, {:.3}",
            self.value, self.wdl.win, self.wdl.draw, self.wdl.loss, self.moves_left
        )
    }
}

//TODO maybe remove this in favour of "parent" and "child"?
impl Flip for ZeroValues {
    fn flip(self) -> Self {
        ZeroValues {
            value: -self.value,
            wdl: self.wdl.flip(),
            moves_left: self.moves_left,
        }
    }
}
