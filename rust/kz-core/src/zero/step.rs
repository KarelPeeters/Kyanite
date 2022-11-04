use std::cmp::Reverse;
use std::fmt::{Display, Formatter};
use std::num::ParseFloatError;
use std::str::FromStr;

use board_game::board::Board;
use board_game::pov::Pov;
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

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum FpuMode {
    Fixed(f32),
    Relative(f32),
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum QMode {
    Value,
    WDL { draw_score: f32 },
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
    q_mode: QMode,
    fpu_root: FpuMode,
    fpu_child: FpuMode,
    virtual_loss: f32,
    rng: &mut impl Rng,
) -> Option<ZeroRequest<B>> {
    let mut curr_node = 0;
    let mut curr_board = tree.root_board().clone();

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
                let mv_count = curr_board.available_moves().count();
                let p = 1.0 / mv_count as f32;

                let start = tree.len();
                curr_board.available_moves().for_each(|mv| {
                    tree.nodes.push(Node::new(Some(curr_node), Some(mv), p));
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

        // go to pov to ensure fixed fpu value is meaningful, quickly convert back to avoid mistakes
        let curr_player = curr_board.next_player();

        // continue selecting
        let selected = if tree[curr_node].complete_visits == 0 {
            // pick a random least-visited child
            choose_max_by_key(children, |&child| Reverse(tree[child].total_visits()), rng)
        } else {
            // pick the best child
            let fpu_mode = if curr_node == 0 { fpu_root } else { fpu_child };

            let uct_context = tree.uct_context(curr_node);
            choose_max_by_key(
                children,
                |&child| {
                    let uct = tree[child]
                        .uct(uct_context, fpu_mode, q_mode, virtual_loss, curr_player)
                        .total(weights);
                    N32::from_inner(uct)
                },
                rng,
            )
        };
        let selected = selected.expect("Board is not done, this node should have a child");

        curr_node = selected;
        curr_board.play(tree[curr_node].last_move.unwrap());
    }
}

/// The second half of a step. Applies a network evaluation to the given node,
/// by setting the child policies and propagating the wdl back to the root.
/// Along the way `virtual_visits` is decremented and `visits` is incremented.
pub fn zero_step_apply<B: Board>(tree: &mut Tree<B>, response: ZeroResponse<B>) {
    // whether we are indeed expecting this node is checked based on (net_values) and (virtual_visits in propagate_values)
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
    for (c, &p) in zip_eq_exact(children, eval.policy.as_ref()) {
        tree[c].net_policy = p;
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
    pub fn select(&self, _parent: ZeroValuesPov) -> ZeroValuesPov {
        todo!("implement again for muzero")
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ModeParseError {
    Prefix(String),
    Float(ParseFloatError),
}

impl Display for FpuMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match *self {
            FpuMode::Fixed(value) => write!(f, "fixed{:+}", value),
            FpuMode::Relative(value) => write!(f, "relative{:+}", value),
        }
    }
}

impl FromStr for FpuMode {
    type Err = ModeParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(rest) = s.strip_prefix("fixed") {
            let value = f32::from_str(rest).map_err(ModeParseError::Float)?;
            return Ok(FpuMode::Fixed(value));
        }

        if let Some(rest) = s.strip_prefix("relative") {
            let value = f32::from_str(rest).map_err(ModeParseError::Float)?;
            return Ok(FpuMode::Relative(value));
        }

        Err(ModeParseError::Prefix(s.to_owned()))
    }
}

impl QMode {
    pub fn wdl() -> QMode {
        QMode::WDL { draw_score: 0.0 }
    }
}

impl Display for QMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match *self {
            QMode::Value => write!(f, "value"),
            QMode::WDL { draw_score } => write!(f, "wdl{:+}", draw_score),
        }
    }
}

impl FromStr for QMode {
    type Err = ModeParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "value" {
            return Ok(QMode::Value);
        }

        if let Some(rest) = s.strip_prefix("wdl") {
            if rest.is_empty() {
                return Ok(QMode::WDL { draw_score: 0.0 });
            } else {
                let value = f32::from_str(rest).map_err(ModeParseError::Float)?;
                return Ok(QMode::WDL { draw_score: value });
            }
        }

        Err(ModeParseError::Prefix(s.to_owned()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fpu_string() {
        assert_eq!(FpuMode::from_str("fixed+0.3"), Ok(FpuMode::Fixed(0.3)));
        assert_eq!(FpuMode::from_str("relative-0.5"), Ok(FpuMode::Relative(-0.5)));

        assert_eq!(&FpuMode::Fixed(0.3).to_string(), "fixed+0.3");
        assert_eq!(&FpuMode::Relative(-0.5).to_string(), "relative-0.5");
    }

    #[test]
    fn q_mode_string() {
        assert_eq!(QMode::from_str("value"), Ok(QMode::Value));
        assert_eq!(QMode::from_str("wdl-0.5"), Ok(QMode::WDL { draw_score: -0.5 }));

        assert_eq!(&QMode::Value.to_string(), "value");
        assert_eq!(&QMode::WDL { draw_score: -0.5 }.to_string(), "wdl-0.5");
    }
}
