use board_game::board::Board;
use board_game::wdl::{Flip, POV, WDL};
use internal_iterator::InternalIterator;

use crate::network::ZeroEvaluation;
use crate::util::zip_eq_exact;
use crate::zero::node::Node;
use crate::zero::range::IdxRange;
use crate::zero::tree::Tree;

#[derive(Debug)]
pub struct ZeroRequest<B> {
    node: usize,
    pub board: B,
}

#[derive(Debug)]
pub struct ZeroResponse {
    node: usize,
    pub eval: ZeroEvaluation,
}

/// The first half of a step, walks down the tree until either:
/// * a **terminal** node is reached.
/// The resulting wdl value is immediately propagated back to the root, the `visit` counters are incremented
/// and `None` is returned.
/// * an **un-evaluated** node is reached.
/// The reached node and its board is returned in a [ZeroRequest],
/// and all involved nodes end up with their `virtual_visits` counter incremented.
///
pub fn zero_step_gather<B: Board>(tree: &mut Tree<B>, exploration_weight: f32) -> Option<ZeroRequest<B>> {
    let mut curr_node = 0;
    let mut curr_board = tree.root_board().clone();

    loop {
        // count each node as visited
        tree[curr_node].virtual_visits += 1;

        // if the board is done backpropagation the real value
        if let Some(outcome) = curr_board.outcome() {
            let wdl = outcome.pov(curr_board.next_player()).to_wdl();
            tree_propagate_wdl(tree, curr_node, wdl);
            return None;
        }

        let children = match tree[curr_node].children {
            None => {
                // initialize the children with uniform policy
                let start = tree.len();
                curr_board.available_moves().for_each(|mv| {
                    tree.nodes.push(Node::new(Some(curr_node), Some(mv), 1.0));
                });
                let end = tree.len();

                tree[curr_node].children = Some(IdxRange::new(start, end));
                tree[curr_node].net_wdl = None;

                // return the request
                return Some(ZeroRequest { board: curr_board, node: curr_node });
            }
            Some(children) => children,
        };

        // continue selecting, pick the best child
        let parent_visits_with_virtual = tree[curr_node].visits_with_virtual();
        let selected = children.iter().max_by_key(|&child| {
            tree[child].uct(exploration_weight, parent_visits_with_virtual)
        }).expect("Board is not done, this node should have a child");

        curr_node = selected;
        curr_board.play(tree[curr_node].last_move.unwrap());
    }
}

/// The second half of a step. Applies a network evaluation to the given node,
/// by setting the child policies and propagating the wdl back to the root.
/// Along the way `virtual_visits` is decremented and `visits` is incremented.
pub fn zero_step_apply<B: Board>(tree: &mut Tree<B>, response: ZeroResponse) {
    // we don't explicitly assert that we're expecting this node since wdl already checks for net_wdl and virtual_visits
    let ZeroResponse { node: curr_node, eval } = response;

    // wdl
    assert!(tree[curr_node].net_wdl.is_none(), "Node {} already has WDL set", curr_node);
    tree[curr_node].net_wdl = Some(eval.wdl);
    tree_propagate_wdl(tree, curr_node, eval.wdl);

    // policy
    let children = tree[curr_node].children.expect("Applied node should have initialized children");
    assert_eq!(children.length as usize, eval.policy.len(), "Wrong children length");
    for (c, p) in zip_eq_exact(children, eval.policy) {
        tree[c].net_policy = p;
    }
}

/// Propagate the given `wdl` up to the root.
fn tree_propagate_wdl<B: Board>(tree: &mut Tree<B>, node: usize, mut wdl: WDL<f32>) {
    let mut curr_index = node;

    loop {
        wdl = wdl.flip();

        let curr_node = &mut tree[curr_index];
        assert!(curr_node.virtual_visits > 0);

        curr_node.visits += 1;
        curr_node.virtual_visits -= 1;
        curr_node.total_wdl += wdl;

        curr_index = match curr_node.parent {
            Some(parent) => parent,
            None => break,
        };
    }
}

impl<B> ZeroRequest<B> {
    pub fn respond(&self, eval: ZeroEvaluation) -> ZeroResponse {
        ZeroResponse { node: self.node, eval }
    }
}