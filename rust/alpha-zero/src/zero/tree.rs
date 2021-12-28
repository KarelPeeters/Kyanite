use std::cmp::{max, min};
use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut};

use board_game::board::{Board, Outcome};
use board_game::wdl::{Flip, OutcomeWDL};
use itertools::Itertools;

use crate::network::ZeroEvaluation;
use crate::util::display_option;
use crate::zero::node::{Node, ZeroValues};
use crate::zero::range::IdxRange;

/// The result of a zero search.
#[derive(Debug, Clone)]
pub struct Tree<B: Board> {
    root_board: B,
    pub(super) nodes: Vec<Node<B::Move>>,
}

impl<B: Board> Tree<B> {
    pub fn new(root_board: B) -> Self {
        assert!(!root_board.is_done(), "Cannot build tree for done board");

        let root = Node::new(None, None, f32::NAN);
        Tree { root_board, nodes: vec![root] }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn root_board(&self) -> &B {
        &self.root_board
    }

    pub fn best_child(&self, node: usize) -> Option<usize> {
        self[node].children.map(|children| {
            children.iter().max_by_key(|&child| {
                (
                    self[child].complete_visits,
                    decorum::Total::from(self[child].net_policy),
                )
            }).unwrap()
        })
    }

    //TODO this doesn't really work for oracle moves any more
    pub fn best_move(&self) -> Option<B::Move> {
        self.best_child(0).map(|c| self[c].last_move.unwrap())
    }

    pub fn principal_variation(&self, max_len: usize) -> Vec<B::Move> {
        std::iter::successors(Some(0), |&n| self.best_child(n))
            .skip(1).take(max_len)
            .map(|n| self[n].last_move.unwrap())
            .collect()
    }

    /// The values corresponding to `root_board` from the POV of `root_board.next_player`.
    pub fn values(&self) -> ZeroValues {
        self[0].values().parent()
    }

    pub fn root_visits(&self) -> u64 {
        self[0].complete_visits
    }

    /// Return `(min, max)` where `min` is the depth of the shallowest un-evaluated node
    /// and `max` is the depth of the deepest evaluated node.
    pub fn depth_range(&self, node: usize) -> (usize, usize) {
        match self[node].children {
            None => (0, 0),
            Some(children) => {
                let mut total_min = usize::MAX;
                let mut total_max = usize::MIN;

                for child in children {
                    let (c_min, c_max) = self.depth_range(child);
                    total_min = min(total_min, c_min);
                    total_max = max(total_max, c_max);
                }

                (total_min + 1, total_max + 1)
            }
        }
    }

    /// Return the policy vector for the root node.
    pub fn policy(&self) -> impl Iterator<Item=f32> + '_ {
        assert!(self.len() > 1, "Must have run for at least 1 iteration");

        self[0].children.unwrap().iter().map(move |c| {
            (self[c].complete_visits as f32) / ((self[0].complete_visits - 1) as f32)
        })
    }

    pub fn eval(&self) -> ZeroEvaluation<'static> {
        ZeroEvaluation {
            values: self.values(),
            policy: self.policy().collect(),
        }
    }

    /// Return a new tree containing the nodes that are still relevant after playing the given move.
    /// Effectively this copies the part of the tree starting from the selected child.
    pub fn keep_move(&self, mv: B::Move) -> Result<Tree<B>, Outcome> {
        //TODO test this function
        assert!(self.len() > 1, "Must have run for at least 1 iteration");

        let mut new_root_board = self.root_board.clone();
        new_root_board.play(mv);
        if let Some(outcome) = new_root_board.outcome() {
            return Err(outcome);
        }

        let picked_child = self[0].children.unwrap().iter()
            .find(|&c| self[c].last_move.unwrap() == mv)
            .unwrap_or_else(|| panic!("Child for move {:?} not found", mv));

        let old_nodes = &self.nodes;
        let mut new_nodes = vec![old_nodes[picked_child].clone()];

        let mut i = 0;

        while i < new_nodes.len() {
            match new_nodes[i].children {
                None => {}
                Some(old_children) => {
                    let new_start = new_nodes.len();
                    new_nodes.extend(old_children.iter().map(|c| old_nodes[c].clone()));
                    let new_end = new_nodes.len();
                    new_nodes[i].children = Some(IdxRange::new(new_start, new_end));
                }
            }

            i += 1;
        }

        let tree = Tree { root_board: new_root_board, nodes: new_nodes };
        Ok(tree)
    }

    #[must_use]
    pub fn display(&self, max_depth: usize, sort: bool, max_children: usize) -> TreeDisplay<B> {
        TreeDisplay { tree: self, node: 0, curr_depth: 0, max_depth, max_children, sort }
    }
}

#[derive(Debug)]
pub struct TreeDisplay<'a, B: Board> {
    tree: &'a Tree<B>,
    node: usize,
    curr_depth: usize,
    max_depth: usize,
    max_children: usize,
    sort: bool,
}

struct PolicyDisplay(f32);

impl Display for PolicyDisplay {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.0 > 0.01 {
            write!(f, "{:.3}", self.0)
        } else {
            write!(f, "{:e}", self.0)
        }
    }
}

impl<B: Board> Display for TreeDisplay<'_, B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let tree = self.tree;

        if self.curr_depth == 0 {
            let data = tree.values();

            writeln!(
                f,
                "values: {}, best_move: {}, depth: {:?}",
                data, display_option(tree.best_move()), tree.depth_range(0)
            )?;
            writeln!(f, "[move: terminal visits zero(v, w/d/l, policy) net(v, w/d/l, policy), uct(q, u)]")?;
        }

        for _ in 0..self.curr_depth { write!(f, "  ")? }

        let node = &self.tree[self.node];
        let parent = node.parent.map(|p| &self.tree[p]);

        let terminal = match node.outcome() {
            Ok(None) => "N",
            Ok(Some(OutcomeWDL::Win)) => "W",
            Ok(Some(OutcomeWDL::Draw)) => "D",
            Ok(Some(OutcomeWDL::Loss)) => "L",
            Err(_) => "?",
        };

        let virtual_visits = if node.virtual_visits != 0 {
            format!("+{}", node.virtual_visits)
        } else {
            String::default()
        };

        let node_values = node.values();
        let net_values = node.net_values.unwrap_or_else(ZeroValues::nan).parent();

        let parent_complete_visits = parent.map_or(node.complete_visits, |p| p.complete_visits);
        let parent_total_visits = parent.map_or(node.total_visits(), |p| p.total_visits());
        let parent_fpu = parent.map_or(ZeroValues::nan(), |p| p.values().flip());

        let zero_policy = if parent_complete_visits > 0 {
            (node.complete_visits as f32) / ((parent_complete_visits - 1) as f32)
        } else {
            f32::NAN
        };

        // TODO use the settings actually used to build the tree here
        let uct = node.uct(parent_total_visits, parent_fpu, false);

        let player = if self.curr_depth % 2 == 0 {
            tree.root_board.next_player().other()
        } else {
            tree.root_board.next_player()
        };

        writeln!(
            f,
            "{} {}: {} {}{} zero({}, {:.4}) net({}, {:.4}) uct({:.4}, {:.4})",
            player.to_char(), display_option(node.last_move), terminal, node.complete_visits, virtual_visits,
            node_values, zero_policy, net_values, node.net_policy,
            uct.q, uct.u,
        )?;

        if self.curr_depth == self.max_depth { return Ok(()); }

        if let Some(children) = node.children {
            let mut children = children.iter().collect_vec();
            let best_child = if self.sort {
                // sort by visits first, then by policy
                children.sort_by_key(|&c| (
                    self.tree[c].complete_visits,
                    decorum::Total::from(self.tree[c].net_policy)
                ));
                children.reverse();
                children[0]
            } else {
                children.iter().copied().max_by_key(|&c| self.tree[c].complete_visits).unwrap()
            };

            for (i, &child) in children.iter().enumerate() {
                if i == self.max_children {
                    for _ in 0..(self.curr_depth + 1) { write!(f, "  ")? }
                    writeln!(f, "...")?;
                    break;
                }

                let next_max_depth = if child == best_child {
                    self.max_depth
                } else {
                    self.curr_depth + 1
                };

                let child_display = TreeDisplay {
                    tree: self.tree,
                    node: child,
                    curr_depth: self.curr_depth + 1,
                    max_depth: next_max_depth,
                    max_children: self.max_children,
                    sort: self.sort,
                };
                write!(f, "{}", child_display)?;
            }
        }

        Ok(())
    }
}

impl<B: Board> Index<usize> for Tree<B> {
    type Output = Node<B::Move>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index]
    }
}

impl<B: Board> IndexMut<usize> for Tree<B> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.nodes[index]
    }
}

