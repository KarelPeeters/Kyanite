use board_game::board::Board;
use board_game::wdl::{Flip, OutcomeWDL, POV};
use decorum::N32;
use internal_iterator::InternalIterator;
use rand::Rng;

use kz_util::sequence::{choose_max_by_key, zip_eq_exact};

use crate::network::ZeroEvaluation;
use crate::zero::node::{Node, UctWeights, ZeroValues};
use crate::zero::range::IdxRange;
use crate::zero::tree::Tree;

#[derive(Debug)]
pub struct ZeroRequest<B> {
    pub node: usize,
    pub board: B,
}

#[derive(Debug)]
pub struct ZeroResponse<'a, B> {
    node: usize,
    pub board: B,
    pub eval: ZeroEvaluation<'a>,
}

#[derive(Debug, Copy, Clone)]
pub enum FpuMode {
    Fixed(ZeroValues),
    Parent,
}

/// The first half of a step, walks down the tree until either:
/// * a **terminal** node is reached.
/// The resulting wdl value is immediately propagated back to the root, the `visit` counters are incremented
/// and `None` is returned.
/// * an **un-evaluated** node is reached.
/// The reached node and its board is returned in a [ZeroRequest],
/// and all involved nodes end up with their `virtual_visits` counter incremented.
///
pub fn zero_step_gather<B: Board>(
    tree: &mut Tree<B>,
    weights: UctWeights,
    use_value: bool,
    fpu_mode: FpuMode,
    rng: &mut impl Rng,
) -> Option<ZeroRequest<B>> {
    let mut curr_node = 0;
    let mut curr_board = tree.root_board().clone();

    //TODO what moves_left to pass here? does it matter?
    //  it's probably better to just switch to q-only fpu
    //  also this while propagating concept may just overcomplicating things
    let mut fpu = ZeroValues::from_outcome(OutcomeWDL::Draw, 0.0);

    loop {
        // count each node as visited
        tree[curr_node].virtual_visits += 1;

        // if the board is done backpropagate the real value
        if let Some(outcome) = curr_board.outcome() {
            let outcome = outcome.pov(curr_board.next_player());
            //TODO what value to use for moves_left?
            tree_propagate_values(tree, curr_node, ZeroValues::from_outcome(outcome, 0.0));
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
                tree[curr_node].net_values = None;

                // return the request
                return Some(ZeroRequest {
                    board: curr_board,
                    node: curr_node,
                });
            }
            Some(children) => children,
        };

        // update fpu
        if tree[curr_node].complete_visits > 0 {
            fpu = tree[curr_node].values();
        }
        //TODO should this be flip or parent? or maybe child?
        fpu = fpu.flip();

        // continue selecting, pick the best child
        let parent_total_visits = tree[curr_node].total_visits();

        let selected = choose_max_by_key(
            children,
            |&child| {
                let uct = tree[child]
                    .uct(parent_total_visits, fpu_mode.select(fpu), use_value)
                    .total(weights);
                N32::from_inner(uct)
            },
            rng,
        )
            .expect("Board is not done, this node should have a child");

        curr_node = selected;
        curr_board.play(tree[curr_node].last_move.unwrap());
    }
}

/// The second half of a step. Applies a network evaluation to the given node,
/// by setting the child policies and propagating the wdl back to the root.
/// Along the way `virtual_visits` is decremented and `visits` is incremented.
pub fn zero_step_apply<B: Board>(tree: &mut Tree<B>, response: ZeroResponse<B>) {
    // we don't explicitly assert that we're expecting this node since wdl already checks for net_wdl and virtual_visits
    let ZeroResponse {
        node: curr_node,
        board: _,
        eval,
    } = response;

    // wdl
    assert!(
        tree[curr_node].net_values.is_none(),
        "Node {} was already evaluated by the network",
        curr_node
    );
    tree[curr_node].net_values = Some(eval.values);
    tree_propagate_values(tree, curr_node, eval.values);

    // policy
    let children = tree[curr_node]
        .children
        .expect("Applied node should have initialized children");
    assert_eq!(children.length as usize, eval.policy.len(), "Wrong children length");
    for (c, &p) in zip_eq_exact(children, eval.policy.as_ref()) {
        tree[c].net_policy = p;
    }
}

/// Propagate the given `wdl` up to the root.
fn tree_propagate_values<B: Board>(tree: &mut Tree<B>, node: usize, mut values: ZeroValues) {
    values = values.flip();
    let mut curr_index = node;

    loop {
        let curr_node = &mut tree[curr_index];
        assert!(curr_node.virtual_visits > 0);

        curr_node.complete_visits += 1;
        curr_node.virtual_visits -= 1;
        curr_node.sum_values += values;

        curr_index = match curr_node.parent {
            Some(parent) => parent,
            None => break,
        };

        values = values.parent();
    }
}

impl FpuMode {
    pub fn select(&self, parent_flipped: ZeroValues) -> ZeroValues {
        match self {
            FpuMode::Fixed(values) => *values,
            FpuMode::Parent => parent_flipped,
        }
    }
}

impl<B> ZeroRequest<B> {
    pub fn respond(self, eval: ZeroEvaluation) -> ZeroResponse<B> {
        ZeroResponse {
            node: self.node,
            board: self.board,
            eval,
        }
    }
}
