use std::convert::TryInto;
use std::num::NonZeroUsize;
use std::ops::{Index, IndexMut};

use itertools::Itertools;
use ordered_float::OrderedFloat;
use sttt::board::{Board, Coord, Player};
use sttt::bot_game::Bot;

use crate::network::Network;

#[derive(Debug, Copy, Clone)]
pub struct IdxRange {
    pub start: NonZeroUsize,
    pub length: u8,
}

impl IdxRange {
    pub fn new(start: usize, end: usize) -> IdxRange {
        IdxRange {
            start: NonZeroUsize::new(start).expect("start cannot be 0"),
            length: (end - start).try_into().expect("length doesn't fit"),
        }
    }

    pub fn iter(&self) -> std::ops::Range<usize> {
        self.start.get()..(self.start.get() + self.length as usize)
    }

    pub fn get(&self, index: usize) -> usize {
        assert!(index < self.length as usize);
        self.start.get() + index
    }
}

impl IntoIterator for IdxRange {
    type Item = usize;
    type IntoIter = std::ops::Range<usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub coord: Coord,

    //this is not just a Option<IdxRange> because of struct layout inefficiencies
    children_start: usize,
    children_length: u8,

    /// The evaluation returned by the network for this position.
    pub evaluation: Option<f32>,
    /// The prior probability as evaluated by the network when the parent node was expanded. Called `P` in the paper.
    pub policy: f32,

    /// The number of times this node has been visited. Called `N` in the paper.
    pub visits: u64,
    /// The sum of final values found in children of this node. Should be divided by `visits` to get the expected value. Called `W` in the paper.
    pub total_value: f32,
}

impl Node {
    fn new(coord: Coord, p: f32) -> Self {
        Node {
            coord,
            children_start: 0,
            children_length: 0,

            evaluation: None,
            policy: p,

            visits: 0,
            total_value: 0.0,
        }
    }

    pub fn value(&self) -> f32 {
        self.total_value / self.visits as f32
    }

    pub fn uct(&self, exploration_weight: f32, parent_visits: u64) -> f32 {
        let q = self.value();
        let u = self.policy * (parent_visits as f32).sqrt() / (1 + self.visits) as f32;
        q + exploration_weight * u
    }

    pub fn children(&self) -> Option<IdxRange> {
        NonZeroUsize::new(self.children_start)
            .map(|start| IdxRange { start, length: self.children_length })
    }

    pub fn set_children(&mut self, children: IdxRange) {
        self.children_start = children.start.get();
        self.children_length = children.length;
    }
}

/// A small wrapper type for Vec<Node> that uses u64 for indexing instead.
#[derive(Debug)]
pub struct Tree {
    root_board: Board,
    nodes: Vec<Node>,
}

impl Tree {
    fn new(root_board: Board) -> Self {
        Tree { root_board, nodes: Default::default() }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn best_move(&self) -> Coord {
        let children = self[0].children()
            .expect("Root node must have children");

        let best_child = children.iter().rev().max_by_key(|&child| {
            self[child].visits
        }).expect("Root node must have non-empty children");

        self[best_child].coord
    }

    fn print_impl(&self, parent_visits: u64, node: usize, depth: usize, max_depth: usize) {
        let node = &self[node];

        for _ in 0..=depth { print!("  ") }
        println!(
            "{:?}: zero({:.3}, {:.3}) net({:.3}, {:.3})", node.coord,
            node.value(), (node.visits as f32) / (parent_visits as f32),
            node.evaluation.unwrap_or(f32::NAN), node.policy,
        );

        if depth == max_depth { return; }

        if let Some(children) = node.children() {
            let best_child = children.start.get() + children.iter()
                .map(|c| OrderedFloat(self[c].value()))
                .position_max().unwrap();

            for child in children {
                let next_max_depth = if child == best_child {
                    max_depth
                } else {
                    depth + 1
                };

                self.print_impl(node.visits, child, depth + 1, next_max_depth)
            }
        }
    }

    pub fn print(&self, max_depth: usize) {
        self.print_impl(self[0].visits, 0, 0, max_depth)
    }

    pub fn keep_move(&self, coord: Coord) -> Tree {
        let mut new_root_board = self.root_board.clone();
        new_root_board.play(coord);

        let picked_child = self[0].children().unwrap().iter()
            .find(|&c| self[c].coord == coord)
            .unwrap();

        let old_nodes = &self.nodes;
        let mut new_nodes = vec![old_nodes[picked_child].clone()];

        let mut i = 0;

        while i < new_nodes.len() {
            match new_nodes[i].children() {
                None => {}
                Some(old_children) => {
                    let new_start = new_nodes.len();
                    new_nodes.extend(old_children.iter().map(|c| old_nodes[c].clone()));
                    let new_end = new_nodes.len();
                    new_nodes[i].set_children(IdxRange::new(new_start, new_end));
                }
            }

            i += 1;
        }

        Tree { root_board: new_root_board, nodes: new_nodes }
    }
}

impl Index<usize> for Tree {
    type Output = Node;

    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index]
    }
}

impl IndexMut<usize> for Tree {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.nodes[index]
    }
}

pub fn mcts_zero_build_tree(board: &Board, iterations: u64, exploration_weight: f32, network: &mut Network) -> Tree {
    assert!(iterations > 0, "MCTS must run for at least 1 iteration");
    assert!(!board.is_done(), "Cannot build MCTS tree for done board");

    let mut tree = Tree::new(board.clone());

    //the actual coord doesn't matter, just pick something
    tree.nodes.push(Node::new(Coord::from_o(0), 1.0));

    mcts_zero_expand_tree(&mut tree, iterations, exploration_weight, network);

    tree
}

pub fn mcts_zero_expand_tree(tree: &mut Tree, iterations: u64, exploration_weight: f32, network: &mut Network) {
    let visits_before = tree[0].visits;

    let mut parent_list = Vec::new();

    for _ in 0..iterations {
        parent_list.clear();

        let mut curr_node: usize = 0;
        let mut curr_board = tree.root_board.clone();

        let mut value = loop {
            parent_list.push(curr_node);

            // if the game is done return the actual value
            if let Some(won_by) = curr_board.won_by {
                let value = if won_by == Player::Neutral { 0.0 } else { -1.0 };
                break value;
            }

            // expand this node if it's the first time and use the network-returned value
            let children = match tree[curr_node].children() {
                None => {
                    //TODO compare with/without value/policy
                    let evaluation = network.evaluate(&curr_board);

                    let start = tree.nodes.len();
                    tree.nodes.extend(curr_board.available_moves().map(|c| {
                        Node::new(c, evaluation.policy[c.o() as usize])
                    }));
                    let length = (tree.nodes.len() - start) as u8;

                    assert!(length > 0);

                    let children = IdxRange {
                        start: NonZeroUsize::new(start).unwrap(),
                        length,
                    };
                    tree[curr_node].set_children(children);

                    tree[curr_node].evaluation = Some(evaluation.value);
                    break evaluation.value;
                }
                Some(children) => children,
            };

            //continue with the best child
            let parent_visits = tree[curr_node].visits;
            let selected = children.iter().max_by_key(|&child| {
                OrderedFloat(tree[child].uct(exploration_weight, parent_visits))
            }).expect("Board is not done, this node should have a child");

            curr_node = selected;
            curr_board.play(tree[curr_node].coord);
        };

        for &update_node in parent_list.iter().rev() {
            value = -value;

            let node = &mut tree[update_node];
            node.visits += 1;
            node.total_value += value;
        }
    }

    assert_eq!(iterations, tree[0].visits - visits_before, "implementation error");
}

pub struct MCTSZeroBot {
    iterations: u64,
    exploration_weight: f32,
    network: Network,
}

impl MCTSZeroBot {
    pub fn new(iterations: u64, exploration_weight: f32, network: Network) -> Self {
        MCTSZeroBot { iterations, exploration_weight, network }
    }

    pub fn build_tree(&mut self, board: &Board) -> Tree {
        mcts_zero_build_tree(board, self.iterations, self.exploration_weight, &mut self.network)
    }
}

impl Bot for MCTSZeroBot {
    fn play(&mut self, board: &Board) -> Option<Coord> {
        if board.is_done() {
            None
        } else {
            let tree = mcts_zero_build_tree(board, self.iterations, self.exploration_weight, &mut self.network);
            Some(tree.best_move())
        }
    }
}