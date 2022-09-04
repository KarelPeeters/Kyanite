use cuda_nn_eval::quant::QuantizedStorage;

use crate::zero::node::Uct;
use crate::zero::range::IdxRange;
use crate::zero::values::ZeroValuesPov;

#[derive(Debug, Clone)]
pub struct MuNode {
    pub parent: Option<usize>,
    pub last_move_index: Option<usize>,

    pub visits: u64,
    pub sum_values: ZeroValuesPov,

    pub net_policy: f32,
    pub inner: Option<MuNodeInner>,
}

#[derive(Debug, Clone)]
pub struct MuNodeInner {
    pub state: QuantizedStorage,
    pub net_values: ZeroValuesPov,
    pub children: IdxRange,
}

impl MuNode {
    pub(super) fn new(parent: Option<usize>, last_move_index: Option<usize>, p: f32) -> Self {
        MuNode {
            parent,
            last_move_index,

            visits: 0,
            sum_values: ZeroValuesPov::default(),

            net_policy: p,
            inner: None,
        }
    }

    /// The (normalized) data of this node from the POV of the player that could play this move.
    /// Does not include virtual visits.
    pub fn values(&self) -> ZeroValuesPov {
        self.sum_values / self.visits as f32
    }

    pub fn uct(&self, parent_total_visits: u64, fpu: ZeroValuesPov, use_value: bool) -> Uct {
        if parent_total_visits == 0 {
            return Uct::nan();
        }

        let total_visits = self.visits;

        let values = if total_visits == 0 { fpu } else { self.values() };

        let v = if use_value {
            values.value.value
        } else {
            values.wdl.value()
        };

        let u = self.net_policy * ((parent_total_visits - 1) as f32).sqrt() / (1 + total_visits) as f32;
        //TODO make sure to remove this -1 if we ever split ZeroValuesPov.flip() into child() and parent()
        let m = values.moves_left - (fpu.moves_left - 1.0);

        Uct { q: v, u, m }
    }
}
