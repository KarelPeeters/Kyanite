use std::cmp::{max, min};
use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut};

use board_game::board::Board;
use board_game::wdl::Flip;
use itertools::Itertools;

use crate::mapping::BoardMapper;
use crate::muzero::node::MuNode;
use kz_util::display_option;

use crate::network::ZeroEvaluation;
use crate::zero::node::ZeroValues;

/// The result of a zero search.
#[derive(Debug)]
pub struct MuTree<B, M> {
    pub(super) root_board: B,
    pub(super) draw_depth: u32,

    pub(super) mapper: M,
    pub(super) nodes: Vec<MuNode>,

    pub(super) current_node_index: Option<usize>,
}

impl<B: Board, M: BoardMapper<B>> MuTree<B, M> {
    pub fn new(root_board: B, draw_depth: u32, mapper: M) -> Self {
        assert!(!root_board.is_done(), "Cannot build tree for done board");
        assert_ne!(
            draw_depth, 0,
            "Draw depth cannot be zero, this would behave like a done board"
        );

        let root_node = MuNode::new(None, None, f32::NAN);
        MuTree {
            root_board,
            draw_depth,
            mapper,
            nodes: vec![root_node],
            current_node_index: None,
        }
    }

    pub fn reserve(&mut self, additional_nodes: usize) {
        self.nodes.reserve(additional_nodes);
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn root_board(&self) -> &B {
        &self.root_board
    }

    pub fn draw_depth(&self) -> u32 {
        self.draw_depth
    }

    pub fn best_child(&self, node: usize) -> Option<usize> {
        self[node].inner.as_ref().map(|inner| {
            inner
                .children
                .iter()
                .max_by_key(|&child| (self[child].visits, decorum::Total::from(self[child].net_policy)))
                .unwrap()
        })
    }

    //TODO change this back to best_move, since root moves are always valid anyway
    pub fn best_move_index(&self) -> Option<usize> {
        self.best_child(0).map(|c| self[c].last_move_index.unwrap())
    }

    pub fn principal_variation(&self, max_len: usize) -> Vec<Option<usize>> {
        std::iter::successors(Some(0), |&n| self.best_child(n))
            .skip(1)
            .take(max_len)
            .map(|n| self[n].last_move_index)
            .collect()
    }

    /// The values corresponding to `root_board` from the POV of `root_board.next_player`.
    pub fn values(&self) -> ZeroValues {
        self[0].values().parent()
    }

    pub fn root_visits(&self) -> u64 {
        self[0].visits
    }

    /// Return `(min, max)` where `min` is the depth of the shallowest un-evaluated node
    /// and `max` is the depth of the deepest evaluated node.
    pub fn depth_range(&self, node: usize) -> (usize, usize) {
        match &self[node].inner {
            None => (0, 0),
            Some(inner) => {
                let mut total_min = usize::MAX;
                let mut total_max = usize::MIN;

                for child in inner.children {
                    let (c_min, c_max) = self.depth_range(child);
                    total_min = min(total_min, c_min);
                    total_max = max(total_max, c_max);
                }

                (total_min + 1, total_max + 1)
            }
        }
    }

    /// Return the policy vector for the root node.
    pub fn policy(&self) -> impl Iterator<Item = f32> + '_ {
        assert!(self.len() > 1, "Must have run for at least 1 iteration");

        self[0]
            .inner
            .as_ref()
            .unwrap()
            .children
            .iter()
            .map(move |c| (self[c].visits as f32) / ((self[0].visits - 1) as f32))
    }
    pub fn eval(&self) -> ZeroEvaluation<'static> {
        ZeroEvaluation {
            values: self.values(),
            policy: self.policy().collect(),
        }
    }
    #[must_use]
    pub fn display(&self, max_depth: usize, sort: bool, max_children: usize, expand_all: bool) -> MuTreeDisplay<B, M> {
        MuTreeDisplay {
            tree: self,
            node: 0,
            curr_depth: 0,
            curr_board: Some(self.root_board.clone()),
            last_mv: None,
            max_depth,
            max_children,
            sort,
            expand_all,
        }
    }
}

#[derive(Debug)]
pub struct MuTreeDisplay<'a, B: Board, M> {
    tree: &'a MuTree<B, M>,
    node: usize,
    curr_depth: usize,
    curr_board: Option<B>,
    last_mv: Option<B::Move>,
    max_depth: usize,
    max_children: usize,
    sort: bool,
    expand_all: bool,
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

impl<B: Board, M: BoardMapper<B>> Display for MuTreeDisplay<'_, B, M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let tree = self.tree;

        if self.curr_depth == 0 {
            let data = tree.values();

            writeln!(
                f,
                "values: {}, best_move_index: {}, depth: {:?}",
                data,
                display_option(tree.best_move_index()),
                tree.depth_range(0)
            )?;
            writeln!(
                f,
                "[move: terminal visits zero({}, p) net({}, p), uct(q, u)]",
                ZeroValues::FORMAT_SUMMARY,
                ZeroValues::FORMAT_SUMMARY
            )?;
        }

        for _ in 0..self.curr_depth {
            write!(f, "  ")?
        }

        let node = &self.tree[self.node];
        let parent = node.parent.map(|p| &self.tree[p]);

        let node_values = node.values();
        let net_values = node
            .inner
            .as_ref()
            .map_or_else(ZeroValues::nan, |inner| inner.net_values)
            .flip();

        let parent_visits = parent.map_or(node.visits, |p| p.visits);
        let parent_fpu = parent.map_or(ZeroValues::nan(), |p| p.values().flip());

        let zero_policy = if parent_visits > 0 {
            (node.visits as f32) / (parent_visits as f32 - 1.0)
        } else {
            f32::NAN
        };

        // TODO use the settings actually used to build the tree here
        let uct = node.uct(parent_visits, parent_fpu, false);

        let player = if self.curr_depth % 2 == 0 {
            tree.root_board.next_player().other()
        } else {
            tree.root_board.next_player()
        };

        writeln!(
            f,
            "{} {}: {} {} zero({}, {:.4}) net({}, {:.4}) {:.4?}",
            player.to_char(),
            display_option(node.last_move_index),
            display_option(self.last_mv),
            node.visits,
            node_values,
            zero_policy,
            net_values,
            node.net_policy,
            uct,
        )?;

        if self.curr_depth == self.max_depth {
            return Ok(());
        }

        if let Some(inner) = &node.inner {
            let mut children = inner.children.iter().collect_vec();
            let best_child = if self.sort {
                // sort by visits first, then by policy
                children.sort_by_key(|&c| (self.tree[c].visits, decorum::Total::from(self.tree[c].net_policy)));
                children.reverse();
                children[0]
            } else {
                children.iter().copied().max_by_key(|&c| self.tree[c].visits).unwrap()
            };

            for (i, &child) in children.iter().enumerate() {
                assert_eq!(tree[child].parent, Some(self.node));

                if i == self.max_children {
                    for _ in 0..(self.curr_depth + 1) {
                        write!(f, "  ")?
                    }
                    writeln!(f, "...")?;
                    break;
                }

                let next_max_depth = if self.expand_all || child == best_child {
                    self.max_depth
                } else {
                    self.curr_depth + 1
                };

                let (child_board, child_mv) = match (&self.curr_board, tree[child].last_move_index) {
                    (Some(curr_board), Some(child_mv_index)) => {
                        let child_mv = tree
                            .mapper
                            .index_to_move(curr_board, child_mv_index)
                            .filter(|&mv| !curr_board.is_done() && curr_board.is_available_move(mv));
                        let child_board = child_mv.map(|child_mv| curr_board.clone_and_play(child_mv));
                        (child_board, child_mv)
                    }
                    _ => (None, None),
                };

                let child_display = MuTreeDisplay {
                    tree: self.tree,
                    node: child,
                    curr_depth: self.curr_depth + 1,
                    curr_board: child_board,
                    last_mv: child_mv,
                    max_depth: next_max_depth,
                    max_children: self.max_children,
                    sort: self.sort,
                    expand_all: self.expand_all,
                };
                write!(f, "{}", child_display)?;
            }
        }

        Ok(())
    }
}

impl<B, M> Index<usize> for MuTree<B, M> {
    type Output = MuNode;

    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index]
    }
}

impl<B, M> IndexMut<usize> for MuTree<B, M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.nodes[index]
    }
}
