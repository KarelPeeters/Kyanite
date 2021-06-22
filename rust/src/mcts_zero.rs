use std::convert::TryInto;
use std::fmt::{Display, Formatter};
use std::num::NonZeroUsize;
use std::ops::{Index, IndexMut};

use itertools::Itertools;
use ordered_float::OrderedFloat;
use sttt::board::{Board, Coord, Player};
use sttt::bot_game::Bot;

use crate::network::{Network, NetworkEvaluation};
use crate::util::EqF32;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct IdxRange {
    pub start: NonZeroUsize,
    pub length: u8,
}

impl IdxRange {
    pub fn new(start: usize, end: usize) -> IdxRange {
        assert!(end > start, "Cannot have empty children");
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

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Node {
    pub coord: Option<Coord>,
    pub children: Option<IdxRange>,

    /// The evaluation returned by the network for this position.
    pub evaluation: Option<EqF32>,
    /// The prior probability as evaluated by the network when the parent node was expanded. Called `P` in the paper.
    pub policy: EqF32,

    /// The number of times this node has been visited. Called `N` in the paper.
    pub visits: u64,
    /// The sum of final values found in children of this node. Should be divided by `visits` to get the expected value. Called `W` in the paper.
    pub total_value: EqF32,
}

impl Node {
    fn new(coord: Option<Coord>, p: f32) -> Self {
        Node {
            coord,
            children: None,

            evaluation: None,
            policy: p.into(),

            visits: 0,
            total_value: 0.0.into(),
        }
    }

    /// The value of this node from the POV of the player that could play this move.
    pub fn value(&self) -> f32 {
        *self.total_value / self.visits as f32
    }

    pub fn uct(&self, exploration_weight: f32, parent_visits: u64) -> f32 {
        let q = self.value();
        let u = *self.policy * (parent_visits as f32).sqrt() / (1 + self.visits) as f32;
        q + exploration_weight * u
    }
}

/// A small wrapper type for Vec<Node> that uses u64 for indexing instead.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Tree {
    root_board: Board,
    nodes: Vec<Node>,
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

pub enum KeepResult {
    Done(Player),
    Tree(Tree),
}

impl Tree {
    pub fn new(root_board: Board) -> Self {
        assert!(!root_board.is_done(), "Cannot build tree for done board");

        let root = Node::new(None, f32::NAN);
        Tree { root_board, nodes: vec![root] }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn root_board(&self) -> &Board {
        &self.root_board
    }

    pub fn best_move(&self) -> Coord {
        assert!(self.len() > 1, "Must have run for at least 1 iteration");

        let children = self[0].children.unwrap();

        let best_child = children.iter().rev().max_by_key(|&child| {
            self[child].visits
        }).expect("Root node must have non-empty children");

        self[best_child].coord.unwrap()
    }

    /// The value of `root_board` from the POV of `root_board.next_player`.
    pub fn value(&self) -> f32 {
        -self[0].value()
    }

    /// Return the policy vector for the root node.
    pub fn policy(&self) -> impl Iterator<Item=f32> + '_ {
        assert!(self.len() > 1, "Must have run for at least 1 iteration");

        self[0].children.unwrap().iter().map(move |c| {
            //TODO isn't this a wrong normalization?
            // the root node is always visited one more time than the sum of the children, right?
            (self[c].visits as f32) / (self[0].visits as f32)
        })
    }

    /// Return a new tree containing the nodes that are still relevant after playing the given move.
    /// Effectively this copies the part of the tree starting from the selected child.
    pub fn keep_move(&self, coord: Coord) -> KeepResult {
        assert!(self.len() > 1, "Must have run for at least 1 iteration");

        let mut new_root_board = self.root_board.clone();
        new_root_board.play(coord);
        if let Some(won_by) = new_root_board.won_by {
            return KeepResult::Done(won_by);
        }

        let picked_child = self[0].children.unwrap().iter()
            .find(|&c| self[c].coord.unwrap() == coord)
            .unwrap_or_else(|| panic!("Child for move {:?} not found", coord));

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
        KeepResult::Tree(tree)
    }

    pub fn display(&self, max_depth: usize) -> TreeDisplay {
        let parent_visits = self[0].visits;
        TreeDisplay { tree: self, node: 0, curr_depth: 0, max_depth, parent_visits }
    }
}

pub struct TreeDisplay<'a> {
    tree: &'a Tree,
    node: usize,
    curr_depth: usize,
    max_depth: usize,
    parent_visits: u64,
}

impl Display for TreeDisplay<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.curr_depth == 0 {
            writeln!(f, "move: visits zero(value, policy) net(value, policy)")?;
        }

        let node = &self.tree[self.node];

        for _ in 0..self.curr_depth { write!(f, "  ")? }
        writeln!(
            f,
            "{:?}: {} zero({:.3}, {:.3}) net({:.3}, {:.3})",
            node.coord, node.visits,
            node.value(), (node.visits as f32) / (self.parent_visits as f32),
            node.evaluation.as_deref().copied().unwrap_or(f32::NAN), node.policy,
        )?;

        if self.curr_depth == self.max_depth { return Ok(()); }

        if let Some(children) = node.children {
            let best_child = children.start.get() + children.iter()
                .map(|c| OrderedFloat(self.tree[c].value()))
                .position_max().unwrap();

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
                    parent_visits: node.visits,
                };
                write!(f, "{}", child_display)?;
            }
        }

        Ok(())
    }
}

/// A coroutine-style implementation that yields `Request`s instead of immediately calling the network.
#[derive(Debug, Clone)]
pub struct ZeroState {
    pub tree: Tree,
    iterations: u64,
    exploration_weight: f32,
    parent_list: Vec<usize>,
}

#[derive(Debug)]
pub enum RunResult {
    Request(Request),
    Done,
}

#[derive(Debug)]
pub struct Request {
    pub board: Board,
    pub node: usize,
}

#[derive(Debug)]
pub struct Response {
    pub request: Request,
    pub evaluation: NetworkEvaluation,
}

impl ZeroState {
    /// Create a new state that will expand the given tree until its root node has been visited `iterations` times.
    pub fn new(tree: Tree, iterations: u64, exploration_weight: f32) -> ZeroState {
        Self { tree, iterations, exploration_weight, parent_list: Vec::with_capacity(81) }
    }

    /// Run until finished or a network evaluation is needed.
    pub fn run_until_result(&mut self, response: Option<Response>) -> RunResult {
        //apply the previous network evaluation if any
        match response {
            None => assert!(self.parent_list.is_empty(), "Expected evaluation response"),
            Some(response) => {
                assert!(!self.parent_list.is_empty(), "Unexpected evaluation response on first run call");
                self.apply_eval(response)
            }
        }

        //continue running
        self.run_until_result_from_root()
    }

    /// Continue running, starting from the selection phase at the root of the tree.
    fn run_until_result_from_root(&mut self) -> RunResult {
        while self.tree[0].visits < self.iterations {
            //start walking down the tree
            assert!(self.parent_list.is_empty());
            let mut curr_node = 0;
            let mut curr_board = self.tree.root_board.clone();

            let value = loop {
                self.parent_list.push(curr_node);

                //if the game is done use the real value
                if let Some(won_by) = curr_board.won_by {
                    let value = if won_by == Player::Neutral { 0.0 } else { -1.0 };
                    break value;
                }

                //get the children or call the network if this is the first time we visit this node
                let children = match self.tree[curr_node].children {
                    None => return RunResult::Request(Request { board: curr_board, node: curr_node }),
                    Some(children) => children,
                };

                //continue with the best child
                let parent_visits = self.tree[curr_node].visits;
                let selected = children.iter().max_by_key(|&child| {
                    OrderedFloat(self.tree[child].uct(self.exploration_weight, parent_visits))
                }).expect("Board is not done, this node should have a child");

                curr_node = selected;
                curr_board.play(self.tree[curr_node].coord.unwrap());
            };

            self.propagate_value(value);
        }

        RunResult::Done
    }

    /// Insert the given network evaluation into the current tree.
    fn apply_eval(&mut self, response: Response) {
        let Response { request, evaluation } = response;
        let Request { board: curr_board, node: curr_node } = request;

        let expected_node = *self.parent_list.last().unwrap();
        assert_eq!(curr_node, expected_node, "Received response for wrong node");

        let start = self.tree.len();
        self.tree.nodes.extend(curr_board.available_moves().map(|c| {
            Node::new(Some(c), evaluation.policy[c.o() as usize])
        }));
        let end = self.tree.len();

        self.tree[curr_node].children = Some(IdxRange::new(start, end));
        self.tree[curr_node].evaluation = Some(evaluation.value.into());

        self.propagate_value(evaluation.value);
    }

    /// Propagate the given final value for a game backwards through the tree using `parent_list`.
    fn propagate_value(&mut self, mut value: f32) {
        assert!(!self.parent_list.is_empty());

        for &node in self.parent_list.iter().rev() {
            value = -value;

            let node = &mut self.tree[node];
            node.visits += 1;
            *node.total_value += value;
        }

        self.parent_list.clear();
    }
}

/// Build a new evaluation tree search from scratch for the given `board`.
pub fn zero_build_tree(board: &Board, iterations: u64, exploration_weight: f32, network: &mut impl Network) -> Tree {
    let mut state = ZeroState::new(Tree::new(board.clone()), iterations, exploration_weight);

    let mut response = None;

    loop {
        let result = state.run_until_result(response);

        match result {
            RunResult::Done => break,
            RunResult::Request(request) => {
                let evaluation = network.evaluate(&request.board);
                response = Some(Response { request, evaluation })
            }
        }
    }

    return state.tree;
}

pub struct ZeroBot<N: Network> {
    iterations: u64,
    exploration_weight: f32,
    network: N,
}

impl<N: Network> ZeroBot<N> {
    pub fn new(iterations: u64, exploration_weight: f32, network: N) -> Self {
        ZeroBot { iterations, exploration_weight, network }
    }

    pub fn build_tree(&mut self, board: &Board) -> Tree {
        zero_build_tree(board, self.iterations, self.exploration_weight, &mut self.network)
    }
}

impl<N: Network> Bot for ZeroBot<N> {
    fn play(&mut self, board: &Board) -> Option<Coord> {
        if board.is_done() {
            None
        } else {
            let tree = zero_build_tree(board, self.iterations, self.exploration_weight, &mut self.network);
            Some(tree.best_move())
        }
    }
}