use std::convert::TryInto;
use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::{Index, IndexMut};

use decorum::N32;
use internal_iterator::InternalIterator;
use itertools::Itertools;
use rand::Rng;
use rand_distr::Distribution;
use board_game::ai::Bot;
use board_game::board::{Board, Outcome};
use board_game::symmetry::{Symmetry, SymmetryDistribution};
use board_game::wdl::{Flip, POV, WDL};

use crate::network::Network;

#[derive(Debug, Copy, Clone)]
pub struct ZeroSettings {
    pub exploration_weight: f32,
    pub random_symmetries: bool,
}

impl ZeroSettings {
    pub fn new(exploration_weight: f32, random_symmetries: bool) -> Self {
        ZeroSettings { exploration_weight, random_symmetries }
    }
}

/// A board evaluation, either as returned by the network or as the final output of a zero tree search.
#[derive(Debug, Clone)]
pub struct ZeroEvaluation {
    /// The win, draw and loss probabilities, after normalization.
    pub wdl: WDL<f32>,

    /// The policy "vector", only containing the available moves in the order they are yielded by `available_moves`.
    pub policy: Vec<f32>,
}

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

#[derive(Debug, Clone)]
pub struct Node<M> {
    pub last_move: Option<M>,
    pub children: Option<IdxRange>,

    /// The evaluation returned by the network for this position.
    pub net_wdl: Option<WDL<f32>>,
    /// The prior probability as evaluated by the network when the parent node was expanded. Called `P` in the paper.
    pub net_policy: f32,

    /// The number of times this node has been visited. Called `N` in the paper.
    pub visits: u64,
    /// The sum of final values found in children of this node. Should be divided by `visits` to get the expected value.
    /// Called `W` in the paper.
    pub total_wdl: WDL<f32>,
}

impl<N> Node<N> {
    fn new(last_move: Option<N>, p: f32) -> Self {
        Node {
            last_move,
            children: None,

            net_wdl: None,
            net_policy: p.into(),

            visits: 0,
            total_wdl: WDL::default(),
        }
    }

    /// The WDL of this node from the POV of the player that could play this move.
    pub fn wdl(&self) -> WDL<f32> {
        //TODO why did we need to change this? did this call never happen for STTT if there were no visits? why?
        if self.visits == 0 {
            WDL::default()
        } else {
            self.total_wdl / self.visits as f32
        }
    }

    pub fn uct(&self, exploration_weight: f32, parent_visits: u64) -> f32 {
        let q = self.wdl().value();
        let u = self.net_policy * (parent_visits as f32).sqrt() / (1 + self.visits) as f32;
        q + exploration_weight * u
    }
}

/// A small wrapper type for Vec<Node> that uses u64 for indexing instead.
#[derive(Debug, Clone)]
pub struct Tree<B: Board> {
    root_board: B,
    nodes: Vec<Node<B::Move>>,
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

#[derive(Debug)]
pub enum KeepResult<B: Board> {
    Done(Outcome),
    Tree(Tree<B>),
}

impl<B: Board> Tree<B> {
    pub fn new(root_board: B) -> Self {
        assert!(!root_board.is_done(), "Cannot build tree for done board");

        let root = Node::new(None, f32::NAN);
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

    /// Return the policy vector for the root node.
    pub fn policy(&self) -> impl Iterator<Item=f32> + '_ {
        assert!(self.len() > 1, "Must have run for at least 1 iteration");

        self[0].children.unwrap().iter().map(move |c| {
            (self[c].visits as f32) / ((self[0].visits - 1) as f32)
        })
    }

    /// Return a new tree containing the nodes that are still relevant after playing the given move.
    /// Effectively this copies the part of the tree starting from the selected child.
    pub fn keep_move(&self, mv: B::Move) -> KeepResult<B> {
        assert!(self.len() > 1, "Must have run for at least 1 iteration");

        let mut new_root_board = self.root_board.clone();
        new_root_board.play(mv);
        if let Some(outcome) = new_root_board.outcome() {
            return KeepResult::Done(outcome);
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
        KeepResult::Tree(tree)
    }

    #[must_use]
    pub fn display(&self, max_depth: usize) -> TreeDisplay<B> {
        let parent_visits = self[0].visits;
        TreeDisplay { tree: self, node: 0, curr_depth: 0, max_depth, parent_visits }
    }
}

#[derive(Debug)]
pub struct TreeDisplay<'a, B: Board> {
    tree: &'a Tree<B>,
    node: usize,
    curr_depth: usize,
    max_depth: usize,
    parent_visits: u64,
}

impl<B: Board> Display for TreeDisplay<'_, B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.curr_depth == 0 {
            writeln!(f, "move: visits zero(w/d/l, policy) net(w/d/l, policy)")?;
        }

        let node = &self.tree[self.node];

        for _ in 0..self.curr_depth { write!(f, "  ")? }

        let node_wdl = node.wdl();
        let net_wdl = node.net_wdl.unwrap_or(WDL::nan()).flip();

        writeln!(
            f,
            "{:?}: {} zero({:.3}/{:.3}/{:.3}, {:.3}) net({:.3}/{:.3}/{:.3}, {:.3})",
            node.last_move, node.visits,
            node_wdl.win, node_wdl.draw, node_wdl.loss,
            (node.visits as f32) / (self.parent_visits as f32),
            net_wdl.win, net_wdl.draw, net_wdl.loss,
            node.net_policy,
        )?;

        if self.curr_depth == self.max_depth { return Ok(()); }

        if let Some(children) = node.children {
            let best_child_index = children.iter()
                .position_max_by_key(|&c| self.tree[c].visits)
                .unwrap();
            let best_child = children.get(best_child_index);

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

/// A coroutine-style implementation that yields `Request`s instead of immediately calling a network.
#[derive(Debug, Clone)]
pub struct ZeroState<B: Board> {
    pub tree: Tree<B>,
    pub target_iterations: u64,
    settings: ZeroSettings,

    parent_list: Vec<usize>,
}

#[derive(Debug)]
pub enum RunResult<B: Board> {
    Request(Request<B>),
    Done,
}

#[derive(Debug, Clone)]
pub struct Request<B: Board> {
    curr_board: B,
    curr_node: usize,
    sym: B::Symmetry,
}

impl<B: Board> Request<B> {
    pub fn board(&self) -> B {
        self.curr_board.map(self.sym)
    }
}

#[derive(Debug)]
pub struct Response<B: Board> {
    pub request: Request<B>,
    pub evaluation: ZeroEvaluation,
}

impl<B: Board> ZeroState<B> {
    /// Create a new state that will expand the given tree until its root node has been visited `iterations` times.
    pub fn new(tree: Tree<B>, target_iterations: u64, settings: ZeroSettings) -> ZeroState<B> {
        Self { tree, target_iterations, settings, parent_list: Vec::with_capacity(81) }
    }

    /// Run until finished or a network evaluation is needed.
    pub fn run_until_result(&mut self, response: Option<Response<B>>, rng: &mut impl Rng) -> RunResult<B> {
        //apply the previous network evaluation if any
        match response {
            None => assert!(self.parent_list.is_empty(), "Expected evaluation response"),
            Some(response) => {
                assert!(!self.parent_list.is_empty(), "Unexpected evaluation response on first run call");
                self.apply_eval(response)
            }
        }

        //continue running
        self.run_until_result_from_root(rng)
    }

    fn gen_symmetry(&self, rng: &mut impl Rng) -> B::Symmetry {
        if self.settings.random_symmetries {
            SymmetryDistribution.sample(rng)
        } else {
            B::Symmetry::identity()
        }
    }

    /// Continue running, starting from the selection phase at the root of the tree.
    fn run_until_result_from_root(&mut self, rng: &mut impl Rng) -> RunResult<B> {
        while self.tree[0].visits < self.target_iterations {
            //start walking down the tree
            assert!(self.parent_list.is_empty());
            let mut curr_node = 0;
            let mut curr_board = self.tree.root_board.clone();

            let wdl = loop {
                self.parent_list.push(curr_node);

                //if the game is done use the real value
                if let Some(outcome) = curr_board.outcome() {
                    break outcome.pov(curr_board.next_player()).to_wdl();
                }

                //get the children or call the network if this is the first time we visit this node
                let children = match self.tree[curr_node].children {
                    None => {
                        let sym = self.gen_symmetry(rng);
                        return RunResult::Request(Request { curr_board, curr_node, sym });
                    }
                    Some(children) => children,
                };

                //continue selecting, pick the best child
                let parent_visits = self.tree[curr_node].visits;
                let selected = children.iter().max_by_key(|&child| {
                    N32::from(self.tree[child].uct(self.settings.exploration_weight, parent_visits))
                }).expect("Board is not done, this node should have a child");

                curr_node = selected;
                curr_board.play(self.tree[curr_node].last_move.unwrap());
            };

            self.propagate_wdl(wdl);
        }

        RunResult::Done
    }

    /// Insert the given network evaluation into the current tree.
    fn apply_eval(&mut self, response: Response<B>) {
        // unwrap everything
        let Response { request, evaluation } = response;
        let ZeroEvaluation { wdl, policy: sym_policy } = evaluation;
        let Request { curr_board, curr_node, sym } = request;

        // safety check: is this actually our request?
        let expected_node = *self.parent_list.last().unwrap();
        assert_eq!(expected_node, curr_node, "Received response for wrong node");

        // store the policy in newly created child nodes while undoing the symmetry map
        let start = self.tree.len();
        for_each_original_move_and_policy(&curr_board, sym, &sym_policy, |mv, p| {
            self.tree.nodes.push(Node::new(Some(mv), p))
        });
        let end = self.tree.len();

        self.tree[curr_node].children = Some(IdxRange::new(start, end));
        self.tree[curr_node].net_wdl = Some(wdl);

        self.propagate_wdl(wdl);
    }

    /// Propagate the given final value for a game backwards through the tree using `parent_list`.
    fn propagate_wdl(&mut self, mut wdl: WDL<f32>) {
        assert!(!self.parent_list.is_empty());

        for &node in self.parent_list.iter().rev() {
            wdl = wdl.flip();

            let node = &mut self.tree[node];
            node.visits += 1;
            node.total_wdl += wdl;
        }

        self.parent_list.clear();
    }
}

/// Visit the available (move, policy) pairs of the given board,
/// assuming sym_policy is the policy evaluated on `board.map(sym)`.
fn for_each_original_move_and_policy<B: Board>(
    board: &B,
    sym: B::Symmetry,
    sym_policy: &Vec<f32>,
    mut f: impl FnMut(B::Move, f32) -> (),
) {
    assert_eq!(board.available_moves().count(), sym_policy.len());

    let policy_sum: f32 = sym_policy.iter().sum();
    assert!((1.0 - policy_sum).abs() < 0.001, "Policy sum was {} != 1.0 for board {}", policy_sum, board);

    //this reverse mapping is kind of ugly but it's probably the best we can do without more constraints on
    // moves and their ordering
    let sym_moves: Vec<B::Move> = board.map(sym).available_moves().collect();

    board.available_moves().for_each(|mv: B::Move| {
        let sym_mv = B::map_move(sym, mv);
        let index = sym_moves.iter().position(|&cand| cand == sym_mv).unwrap();
        f(mv, sym_policy[index])
    });
}

/// Build a new evaluation tree search from scratch for the given `board`.
pub fn zero_build_tree<B: Board>(
    board: &B,
    iterations: u64,
    settings: ZeroSettings,
    network: &mut impl Network<B>,
    rng: &mut impl Rng,
) -> Tree<B> {
    let mut state = ZeroState::new(Tree::new(board.clone()), iterations, settings);

    let mut response = None;

    loop {
        let result = state.run_until_result(response, rng);

        match result {
            RunResult::Done => break,
            RunResult::Request(request) => {
                let evaluation = network.evaluate(&request.board());
                response = Some(Response { request, evaluation })
            }
        }
    }

    return state.tree;
}

pub struct ZeroBot<B: Board, N: Network<B>, R: Rng> {
    iterations: u64,
    settings: ZeroSettings,
    network: N,
    rng: R,
    ph: PhantomData<*const B>,
}

impl<B: Board, N: Network<B>, R: Rng> Debug for ZeroBot<B, N, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ZeroBot {{ iterations: {:?}, settings: {:?}, network: {:?} }}", self.iterations, self.settings, self.network)
    }
}

impl<B: Board, N: Network<B>, R: Rng> ZeroBot<B, N, R> {
    pub fn new(iterations: u64, settings: ZeroSettings, network: N, rng: R) -> Self {
        ZeroBot { iterations, settings, network, rng, ph: PhantomData }
    }

    /// Utility function that builds a tree with the settings of this bot.
    pub fn build_tree(&mut self, board: &B) -> Tree<B> {
        zero_build_tree(board, self.iterations, self.settings, &mut self.network, &mut self.rng)
    }
}

impl<B: Board, N: Network<B>, R: Rng> Bot<B> for ZeroBot<B, N, R> {
    fn select_move(&mut self, board: &B) -> B::Move {
        self.build_tree(board).best_move()
    }
}