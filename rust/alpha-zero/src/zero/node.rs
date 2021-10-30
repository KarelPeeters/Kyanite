use board_game::wdl::{OutcomeWDL, WDL};
use decorum::N32;

use crate::zero::range::IdxRange;

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
    pub visits: u64,
    /// The number of virtual visits for this node and its children.
    pub virtual_visits: u64,
    /// The sum of final values found in children of this node. Should be divided by `visits` to get the expected value.
    pub total_wdl: WDL<f32>,

    /// The evaluation returned by the network for this position.
    /// If `None` and the node has children this means this node currently has virtual WDL applied to it.
    pub net_wdl: Option<WDL<f32>>,
    /// The policy/prior probability as evaluated by the network when the parent node was expanded.
    pub net_policy: f32,
}

impl<N> Node<N> {
    pub(super) fn new(parent: Option<usize>, last_move: Option<N>, p: f32) -> Self {
        Node {
            parent,
            last_move,
            children: None,

            visits: 0,
            virtual_visits: 0,
            total_wdl: WDL::default(),

            net_wdl: None,
            net_policy: p.into(),
        }
    }

    /// The (normalized) WDL of this node from the POV of the player that could play this move.
    /// Does not include virtual loss.
    pub fn wdl(&self) -> WDL<f32> {
        self.total_wdl / self.total_wdl.sum()
    }

    /// Get the outcome of this node if it's terminal.
    /// * `Err` means we don't know yet because this node has not been visited yet,
    /// * `Ok(None)` means this node is not terminal.
    /// * `Some(outcome) ` is the outcome of this node
    pub fn terminal(&self) -> Result<Option<OutcomeWDL>, ()> {
        if self.children.is_none() {
            if self.visits_with_virtual() > 0 {
                let outcome = self.total_wdl.try_to_outcome_wdl()
                    .unwrap_or_else(|()| panic!("Unexpected wdl {:?} for terminal node", self.total_wdl));
                Ok(Some(outcome))
            } else {
                Err(())
            }
        } else {
            Ok(None)
        }
    }

    pub(super) fn visits_with_virtual(&self) -> u64 {
        self.visits + self.virtual_visits
    }

    pub(super) fn total_wdl_with_virtual(&self) -> WDL<f32> {
        self.total_wdl + WDL::new(0.0, 0.0, self.virtual_visits as f32)
    }

    pub(super) fn uct(&self, exploration_weight: f32, parent_visits_with_virtual: u64) -> N32 {
        let visits_with_virtual = self.visits_with_virtual();
        let total_wdl_with_virtual = self.total_wdl_with_virtual();

        let v = if visits_with_virtual == 0 {
            0.0
        } else {
            total_wdl_with_virtual.value() / visits_with_virtual as f32
        };

        let q = (v + 1.0) / 2.0;
        let u = self.net_policy * ((parent_visits_with_virtual - 1) as f32).sqrt() / (1 + visits_with_virtual) as f32;

        N32::from(q + exploration_weight * u)
    }
}
