use board_game::board::{Board, Outcome};
use board_game::pov::{NonPov, Pov};
use decorum::N32;
use internal_iterator::InternalIterator;
use rand::Rng;

use kz_util::sequence::{choose_max_by_key, zip_eq_exact};

use crate::network::ZeroEvaluation;
use crate::zero::node::{Node, UctWeights};
use crate::zero::range::IdxRange;
use crate::zero::tree::Tree;
use crate::zero::values::{ZeroValuesAbs, ZeroValuesPov};

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
    Fixed(ZeroValuesPov),
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
    let mut last_parent_values = ZeroValuesAbs::from_outcome(Outcome::Draw, 0.0);

    loop {
        // count each node as visited
        tree[curr_node].virtual_visits += 1;

        // if the board is done backpropagate the real value
        if let Some(outcome) = curr_board.outcome() {
            tree_propagate_values(tree, curr_node, ZeroValuesAbs::from_outcome(outcome, 0.0));
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

        // update fpu if we have any information
        // TODO consider using a bigger threshold here?
        // TODO should we set fpu to zero again if we have no information? -> no (?)
        if tree[curr_node].complete_visits > 0 {
            last_parent_values = tree[curr_node].values();
        }

        // go to pov to ensure fixed fpu value is meaningful, quickly convert back to avoid mistakes
        let curr_player = curr_board.next_player();
        let used_fpu = fpu_mode.select(last_parent_values.pov(curr_player)).un_pov(curr_player);

        // continue selecting, pick the best child
        let parent_total_visits = tree[curr_node].total_visits();
        let selected = choose_max_by_key(
            children,
            |&child| {
                let uct = tree[child]
                    .uct(
                        parent_total_visits,
                        last_parent_values.moves_left,
                        used_fpu,
                        use_value,
                        curr_player,
                    )
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
pub fn zero_step_apply<B: Board>(tree: &mut Tree<B>, policy_temperature: f32, response: ZeroResponse<B>) {
    // whether we are indeed expecting this node is checked based on (net_values) and (virtual visits in propagate_values)
    let ZeroResponse {
        node: curr_node,
        board: curr_board,
        eval,
    } = response;
    let curr_player = curr_board.next_player();

    // values
    assert!(
        tree[curr_node].net_values.is_none(),
        "Node {} was already evaluated by the network",
        curr_node
    );
    let values_abs = eval.values.un_pov(curr_player);
    tree[curr_node].net_values = Some(values_abs);
    tree_propagate_values(tree, curr_node, values_abs);

    // policy
    let children = tree[curr_node]
        .children
        .expect("Applied node should have initialized children");
    assert_eq!(children.length as usize, eval.policy.len(), "Wrong children length");

    //  set policies, apply temperature
    let mut total_p_temp = 0.0;
    for (c, &p) in zip_eq_exact(children, eval.policy.as_ref()) {
        let p_temp = p.powf(1.0 / policy_temperature);
        total_p_temp += p_temp;
        tree[c].net_policy = p_temp;
    }

    //  renormalize policy
    assert!(total_p_temp > 0.0);
    for c in children {
        tree[c].net_policy /= total_p_temp;
    }
}

/// Propagate the given `wdl` up to the root.
fn tree_propagate_values<B: Board>(tree: &mut Tree<B>, node: usize, mut values: ZeroValuesAbs) {
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
    pub fn select(&self, parent: ZeroValuesPov) -> ZeroValuesPov {
        match self {
            FpuMode::Fixed(values) => *values,
            FpuMode::Parent => parent,
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
