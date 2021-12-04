use std::fmt::{Display, Formatter};

use board_game::wdl::{Flip, OutcomeWDL, WDL};
use decorum::N32;

use crate::zero::range::IdxRange;

/// The data that is accumulated in a node.
#[derive(Debug, Copy, Clone, Default)]
pub struct ZeroValues {
    pub value: f32,
    pub wdl: WDL<f32>,
}

#[derive(Debug, Clone)]
// TODO look at the size of this struct and think about making it smaller
//   (but first try padding it so see if that makes it slower)
pub struct Node<M> {
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
            net_policy: p.into(),
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
    /// * `Err(())` means we don't know yet because this node has not been visited yet,
    /// * `Ok(None)` means this node is not terminal.
    /// * `Ok(Some(outcome))` is the outcome of this node
    pub fn outcome(&self) -> Result<Option<OutcomeWDL>, ()> {
        if self.children.is_none() {
            if self.total_visits() > 0 {
                let outcome = self.total_data().wdl.try_to_outcome_wdl()
                    .unwrap_or_else(|()| panic!("Unexpected wdl {:?} for terminal node", self.total_data().wdl));
                Ok(Some(outcome))
            } else {
                Err(())
            }
        } else {
            Ok(None)
        }
    }

    pub(super) fn uct(&self, parent_total_visits: u64, fpu: ZeroValues, exploration_weight: f32, use_value: bool) -> N32 {
        let total_visits = self.total_visits();

        let data = if total_visits == 0 {
            fpu
        } else {
            self.total_data()
        };

        let v = if use_value {
            data.value
        } else {
            data.wdl.value()
        };

        let q = (v + 1.0) / 2.0;
        let u = self.net_policy * ((parent_total_visits - 1) as f32).sqrt() / (1 + total_visits) as f32;

        N32::from(q + exploration_weight * u)
    }
}

impl ZeroValues {
    pub fn from_outcome(outcome: OutcomeWDL) -> Self {
        ZeroValues {
            value: outcome.sign(),
            wdl: outcome.to_wdl(),
        }
    }

    pub fn nan() -> Self {
        ZeroValues { value: f32::NAN, wdl: WDL::nan() }
    }

    /// The value that should be accumulated in the parent node of this value.
    /// This is similar to [WDL::flip] but in the future will also do things like increment the _moves left counter_.
    pub fn parent(&self) -> Self {
        ZeroValues { value: -self.value, wdl: self.wdl.flip() }
    }

    pub fn add_virtual(&self, virtual_visits: u64) -> Self {
        let virtual_visits = virtual_visits as f32;
        ZeroValues {
            value: self.value + virtual_visits,
            wdl: self.wdl + WDL::new(0.0, 0.0, virtual_visits),
        }
    }
}

impl std::ops::Add<Self> for ZeroValues {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        ZeroValues { value: self.value + rhs.value, wdl: self.wdl + rhs.wdl }
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
        ZeroValues { value: self.value / rhs, wdl: self.wdl / rhs }
    }
}

impl Display for ZeroValues {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.3}, {:.3}/{:.3}/{:.3}", self.value, self.wdl.win, self.wdl.draw, self.wdl.loss)
    }
}

impl Flip for ZeroValues {
    fn flip(self) -> Self {
        ZeroValues {
            value: -self.value,
            wdl: self.wdl.flip(),
        }
    }
}