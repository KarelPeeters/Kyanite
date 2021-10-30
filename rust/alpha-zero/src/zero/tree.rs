use std::cmp::{max, min};
use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut};

use board_game::board::{Board, Outcome};
use board_game::wdl::{Flip, OutcomeWDL, WDL};
use itertools::Itertools;

use crate::util::display_option;
use crate::zero::node::Node;
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

    pub fn best_move(&self) -> B::Move {
        assert!(self.len() > 1, "Must have run for at least 1 iteration");

        let children = self[0].children.unwrap();

        let best_child = children.iter().rev().max_by_key(|&child| {
            self[child].visits
        }).expect("Root node must have non-empty children");

        self[best_child].last_move.unwrap()
    }

    /// The WDL of `root_board` from the POV of `root_board.next_player`.
    pub fn wdl(&self) -> WDL<f32> {
        self[0].wdl().flip()
    }

    pub fn root_visits(&self) -> u64 {
        self[0].visits
    }

    /// Return `(min, max)` where `min` is the depth of the shallowest un-evaluated node
    /// and `max` is the depth of the deepest evaluated node.
    pub fn depth_range(&self, start: usize) -> (usize, usize) {
        match self[start].children {
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
            (self[c].visits as f32) / ((self[0].visits - 1) as f32)
        })
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
    pub fn display(&self, max_depth: usize, sort: bool) -> TreeDisplay<B> {
        let parent_visits = self[0].visits;
        TreeDisplay { tree: self, node: 0, curr_depth: 0, max_depth, sort, parent_visits }
    }
}

#[derive(Debug)]
pub struct TreeDisplay<'a, B: Board> {
    tree: &'a Tree<B>,
    node: usize,
    curr_depth: usize,
    max_depth: usize,
    sort: bool,
    parent_visits: u64,
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
            let wdl = tree.wdl();
            writeln!(
                f,
                "wdl: ({:.3}/{:.3}/{:.3}), best move: {}, depth: {:?}",
                wdl.win, wdl.draw, wdl.loss, tree.best_move(), tree.depth_range(0)
            )?;
            writeln!(f, "[move: terminal visits zero(w/d/l, policy) net(w/d/l, policy)]")?;
        }

        for _ in 0..self.curr_depth { write!(f, "  ")? }

        let node = &self.tree[self.node];
        let node_wdl = node.wdl();
        let net_wdl = node.net_wdl.unwrap_or(WDL::nan()).flip();

        let terminal = match node.terminal() {
            Ok(None) => "N",
            Ok(Some(OutcomeWDL::Win)) => "W",
            Ok(Some(OutcomeWDL::Draw)) => "D",
            Ok(Some(OutcomeWDL::Loss)) => "L",
            Err(()) => "?",
        };

        let virtual_visits = if node.virtual_visits != 0 {
            format!("+{}", node.virtual_visits)
        } else {
            String::default()
        };

        writeln!(
            f,
            "{}: {} {}{} zero({:.3}/{:.3}/{:.3}, {:.4}) net({:.3}/{:.3}/{:.3}, {:.4})",
            display_option(node.last_move), terminal, node.visits, virtual_visits,
            node_wdl.win, node_wdl.draw, node_wdl.loss,
            (node.visits as f32) / (self.parent_visits as f32),
            net_wdl.win, net_wdl.draw, net_wdl.loss,
            node.net_policy,
        )?;

        if self.curr_depth == self.max_depth { return Ok(()); }

        if let Some(children) = node.children {
            let mut children = children.iter().collect_vec();
            let best_child = if self.sort {
                children.sort_by_key(|&c| self.tree[c].visits);
                children.reverse();
                children[0]
            } else {
                children.iter().copied().max_by_key(|&c| self.tree[c].visits).unwrap()
            };

            for child in children {
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
                    sort: self.sort,
                    parent_visits: node.visits,
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

