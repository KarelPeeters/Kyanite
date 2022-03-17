use board_game::board::Board;
use board_game::wdl::{Flip, OutcomeWDL};
use cuda_nn_eval::tensor::DeviceTensor;
use decorum::N32;
use itertools::Itertools;

use crate::mapping::BoardMapper;
use crate::muzero::node::{MuNode, MuNodeInner};
use crate::muzero::tree::MuTree;
use crate::muzero::MuZeroEvaluation;
use crate::zero::node::{UctWeights, ZeroValues};
use crate::zero::range::IdxRange;
use crate::zero::step::FpuMode;

#[derive(Debug)]
pub enum MuZeroRequest<B> {
    Root {
        node: usize,
        board: B,
    },
    Expand {
        node: usize,
        state: DeviceTensor,
        move_index: usize,
    },
}

#[derive(Debug)]
pub struct MuZeroResponse<'a> {
    pub node: usize,
    pub state: DeviceTensor,
    pub eval: MuZeroEvaluation<'a>,
}

pub fn muzero_step_gather<B: Board, M: BoardMapper<B>>(
    tree: &MuTree<B, M>,
    weights: UctWeights,
    use_value: bool,
    fpu_mode: FpuMode,
) -> Option<MuZeroRequest<B>> {
    if tree[0].inner.is_none() {
        return Some(MuZeroRequest::Root {
            node: 0,
            board: tree.root_board().clone(),
        });
    }

    let mut curr_node = 0;
    let mut fpu = ZeroValues::from_outcome(OutcomeWDL::Draw, 0.0);

    let mut last_move_index = None;
    let mut last_state: Option<DeviceTensor> = None;

    loop {
        let inner = if let Some(inner) = &tree[curr_node].inner {
            inner
        } else {
            return Some(MuZeroRequest::Expand {
                node: curr_node,
                state: last_state.unwrap(),
                move_index: last_move_index.unwrap(),
            });
        };

        // update fpu
        if tree[curr_node].visits > 0 {
            fpu = tree[curr_node].values();
        }
        //TODO should this be flip or parent? or maybe child?
        fpu = fpu.flip();

        // continue selecting, pick the best child
        let parent_total_visits = tree[curr_node].visits;

        let selected_index = inner
            .children
            .iter()
            .position_max_by_key(|&child| {
                let x = tree[child]
                    .uct(parent_total_visits, fpu_mode.select(fpu), use_value)
                    .total(weights);
                N32::from_inner(x)
            })
            .expect("Children cannot be be empty");

        let selected = inner.children.get(selected_index);

        curr_node = selected;

        last_move_index = Some(selected_index);
        last_state = Some(inner.state.clone());
    }
}

/// The second half of a step. Applies a network evaluation to the given node,
/// by setting the child policies and propagating the wdl back to the root.
/// Along the way `virtual_visits` is decremented and `visits` is incremented.
pub fn muzero_step_apply<B: Board, M: BoardMapper<B>>(tree: &mut MuTree<B, M>, response: MuZeroResponse) {
    let MuZeroResponse {
        node,
        state,
        eval: MuZeroEvaluation { values, policy },
    } = response;

    assert_eq!(tree.mapper.policy_len(), policy.len(), "Mismatching policy length");

    // create children with correct policy
    let start = tree.nodes.len();
    for (i, &p) in policy.as_ref().iter().enumerate() {
        tree.nodes.push(MuNode::new(Some(node), Some(i), p))
    }
    let end = tree.nodes.len();

    // set node inner
    let inner = MuNodeInner {
        state,
        net_values: values,
        children: IdxRange::new(start, end),
    };
    tree[node].inner = Some(inner);

    // propagate values
    tree_propagate_values(tree, node, values);
}

/// Propagate the given `values` up to the root.
fn tree_propagate_values<B: Board, M: BoardMapper<B>>(tree: &mut MuTree<B, M>, node: usize, mut values: ZeroValues) {
    values = values.flip();
    let mut curr_index = node;

    loop {
        let curr_node = &mut tree[curr_index];

        curr_node.visits += 1;
        curr_node.sum_values += values;

        curr_index = match curr_node.parent {
            Some(parent) => parent,
            None => break,
        };

        values = values.parent();
    }
}
