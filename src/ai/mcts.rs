use std::fmt::{Debug, Formatter};
use std::num::NonZeroUsize;
use std::ops::{Index, IndexMut};

use decorum::N32;
use internal_iterator::InternalIterator;
use rand::Rng;
use rand::seq::IteratorRandom;

use crate::ai::Bot;
use crate::board::{Board, Outcome};
use crate::wdl::{Flip, OutcomeWDL, POV, WDL};

#[derive(Debug, Copy, Clone)]
pub struct IdxRange {
    pub start: NonZeroUsize,
    pub length: usize,
}

impl IdxRange {
    pub fn iter(&self) -> std::ops::Range<usize> {
        let start = self.start.get();
        start..(start + self.length)
    }

    pub fn get(&self, index: usize) -> usize {
        assert!(index < (self.length as usize), "Index {} out of bounds", index);
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

/// Represents a node in the MCTS search tree.
///
/// The outcome or wdl in this node are always from the POV of the player that just played `self.last_move`.
#[derive(Debug)]
pub struct Node<M> {
    pub last_move: Option<M>,
    pub children: Option<IdxRange>,
    pub visits: i64,
    pub kind: SNodeKind,
}

#[derive(Debug)]
pub enum SNodeKind {
    Estimate(WDL<i64>),
    Solved(OutcomeWDL),
}

impl<M> Node<M> {
    /// Create the correct type of `Node` for the given `outcome`.
    /// ´outcome` should be from the POV of the player that just played `last_move`.
    fn new(last_move: Option<M>, outcome: Option<OutcomeWDL>) -> Self {
        let kind = match outcome {
            None => SNodeKind::Estimate(WDL::default()),
            Some(outcome) => SNodeKind::Solved(outcome),
        };

        Node { last_move, visits: 0, children: None, kind }
    }

    pub fn is_unvisited(&self) -> bool {
        match self.kind {
            SNodeKind::Estimate(wdl) => wdl.sum() == 0,
            SNodeKind::Solved(_) => false,
        }
    }

    /// Return the solution (if any) from the POV of the player that just played `self.last_move`.
    pub fn solution(&self) -> Option<OutcomeWDL> {
        match self.kind {
            SNodeKind::Estimate(_) => None,
            SNodeKind::Solved(outcome) => Some(outcome),
        }
    }

    pub fn mark_solved(&mut self, outcome: OutcomeWDL) {
        self.visits += 1;
        if let SNodeKind::Solved(_) = self.kind {
            panic!("Cannot mark already solved node as solved again");
        }

        self.kind = SNodeKind::Solved(outcome);
    }

    pub fn increment(&mut self, outcome: OutcomeWDL) {
        self.visits += 1;
        match &mut self.kind {
            SNodeKind::Estimate(wdl) => {
                *wdl += outcome.to_wdl();
            }
            SNodeKind::Solved(_) => {
                panic!("Cannot increment solved node")
            }
        }
    }

    /// The value of this node from the POV of the player that just played `self.last_move`.
    pub fn wdl(&self) -> WDL<f32> {
        match self.kind {
            SNodeKind::Estimate(wdl) => {
                let visits = wdl.sum();
                wdl.cast::<f32>() / visits as f32
            }
            SNodeKind::Solved(outcome) => {
                outcome.to_wdl()
            }
        }
    }

    /// Return the uct value of this code.
    ///
    /// For solved nodes this is just the unit value, with no exploration bonus. This is equivalent to
    /// a child node that's visited an infinite amount of times (together with the parent node).
    fn uct(&self, parent_visits: i64, exploration_weight: f32) -> f32 {
        match self.kind {
            SNodeKind::Estimate(wdl) => {
                let visits = wdl.sum() as f32;
                let value = wdl.cast::<f32>().value() / visits;
                let value_unit = (value + 1.0) / 2.0;

                let explore = ((parent_visits as f32).ln() / visits).sqrt();

                value_unit + exploration_weight * explore
            }
            SNodeKind::Solved(outcome) => {
                (outcome.sign::<f32>() + 1.0) / 2.0
            }
        }
    }
}

/// A small wrapper type for Vec<SNode> that uses u64 for indexing instead.
#[derive(Debug)]
pub struct Tree<B: Board> {
    pub root_board: B,
    pub nodes: Vec<Node<B::Move>>,
}

impl<B: Board> Tree<B> {
    pub fn new(root_board: B) -> Self {
        Tree { root_board, nodes: Default::default() }
    }

    pub fn best_child(&self) -> usize {
        let children = self[0].children
            .expect("Root node must have children");

        //pick the winning child if any
        // there should only be at most one, so we're not biasing towards earlier moves here
        let won_child = children.iter()
            .find(|&c| self[c].solution() == Some(OutcomeWDL::Win));
        if let Some(win_child) = won_child {
            return win_child;
        }

        // pick the most visited child
        children.iter()
            .max_by_key(|&c| self[c].visits)
            .unwrap()
    }

    pub fn best_move(&self) -> B::Move {
        let best_child = self.best_child();
        self[best_child].last_move.unwrap()
    }

    /// The wdl of `root_board` from the POV of `root_board.next_player`.
    pub fn wdl(&self) -> WDL<f32> {
        // the evaluation of the starting board is the opposite of the the evaluation of the root node,
        //   since that is the evaluation from the POV of the "previous" player
        self[0].wdl().flip()
    }

    pub fn print(&self, depth: u64) {
        println!("move: visits, value <- W,D,L");
        self.print_impl(0, 0, depth);
        println!();
    }

    fn print_impl(&self, node: usize, depth: u64, max_depth: u64) {
        let node = &self[node];

        for _ in 0..depth { print!("  ") }
        print!("{:?}: {}, {:.3} <- ", node.last_move, node.visits, node.wdl().value());

        match node.kind {
            SNodeKind::Estimate(_) => {
                let wdl = node.wdl();
                println!("{:.3},{:.3},{:.3}", wdl.win, wdl.draw, wdl.loss);
            }
            SNodeKind::Solved(outcome) => {
                println!("{:?}", outcome);
            }
        };

        if depth == max_depth { return; }

        if let Some(children) = node.children {
            let best_child = self.best_child();

            for child in children {
                let next_max_depth = if child == best_child {
                    max_depth
                } else {
                    depth + 1
                };

                self.print_impl(child, depth + 1, next_max_depth)
            }
        }
    }
}

impl<B: Board> Index<usize> for Tree<B> {
    type Output = Node<B::Move>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index as usize]
    }
}

impl<B: Board> IndexMut<usize> for Tree<B> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.nodes[index as usize]
    }
}

fn random_playout<B: Board>(mut board: B, rng: &mut impl Rng) -> Outcome {
    assert!(!board.is_done(), "should never start random playout on a done board");

    loop {
        board.play(board.random_available_move(rng));

        if let Some(outcome) = board.outcome() {
            return outcome;
        }
    }
}

/// Run a single MCTS step.
///
/// Returns `(result, proven)`, where
/// * `result` is from the pov of the player that just played on `curr_board`.
/// * `proven` is whether this result is fully proven
///
/// This function has already increments `curr_node` before it returns.
fn mcts_solver_step<B: Board>(
    tree: &mut Tree<B>,
    curr_node: usize,
    curr_board: &B,
    exploration_weight: f32,
    rng: &mut impl Rng,
) -> (OutcomeWDL, bool) {
    //TODO should we decrement visit count? -> meh, then we're pulling search time towards partially solved branches
    //TODO should we backprop all previous backpropped losses and draws as wins now? -> meh, then we're overestimating this entire branch

    if let Some(outcome) = tree[curr_node].solution() {
        return (outcome, true);
    }

    // initialize children
    let children = match tree[curr_node].children {
        Some(children) => children,
        None => {
            let start = NonZeroUsize::new(tree.nodes.len()).unwrap();

            curr_board.available_moves().for_each(|mv: B::Move| {
                let next_board = curr_board.clone_and_play(mv);
                let outcome = next_board.outcome().pov(curr_board.next_player());
                let node = Node::new(Some(mv), outcome);
                tree.nodes.push(node);
            });

            let length = tree.nodes.len() - start.get();
            let children = IdxRange { start, length };
            tree[curr_node].children = Some(children);

            //TODO maybe do this even earlier, and immediately stop pushing nodes -> but then children are inconsistent :(
            //  so what? who care about children somewhere deep in the tree!
            let outcome = OutcomeWDL::best(children.iter().map(|c| tree[c].solution()));
            if let Some(outcome) = outcome.flip() {
                tree[curr_node].mark_solved(outcome);
                return (outcome, true);
            } else {
                children
            }
        }
    };

    // check if there are unvisited children
    let unvisited = children.iter().filter(|&c| tree[c].is_unvisited());
    let picked_unvisited = unvisited.choose(rng);

    // result is from the POV of curr_board.next_player
    let (result, proven) = if let Some(picked_child) = picked_unvisited {
        let picked_mv = tree[picked_child].last_move.unwrap();
        let next_board = curr_board.clone_and_play(picked_mv);

        let outcome = random_playout(next_board, rng)
            .pov(curr_board.next_player().other());
        tree[picked_child].increment(outcome);

        (outcome.flip(), false)
    } else {
        //pick the max-uct child
        //TODO we're including lost and drawn nodes here, is there nothing better we can do?
        // at least this is what the paper seems to suggest
        let parent_visits = tree[curr_node].visits;

        let picked = children.iter()
            .max_by_key(|&c| {
                N32::from(tree[c].uct(parent_visits, exploration_weight))
            })
            .unwrap();

        //continue recursing
        let picked_mv = tree[picked].last_move.unwrap();
        let next_board = curr_board.clone_and_play(picked_mv);

        mcts_solver_step(tree, picked, &next_board, exploration_weight, rng)
    };

    let result = result.flip();

    if proven {
        //check if we can prove the current node as well
        let outcome = OutcomeWDL::best(children.iter().map(|c| tree[c].solution()));
        if let Some(outcome) = outcome.flip() {
            tree[curr_node].mark_solved(outcome);
            return (outcome, true);
        }
    }

    tree[curr_node].increment(result);
    (result, false)
}

pub fn mcts_build_tree<B: Board>(root_board: &B, iterations: u64, exploration_weight: f32, rng: &mut impl Rng) -> Tree<B> {
    assert!(iterations > 0);

    let mut tree = Tree::new(root_board.clone());

    let root_outcome = root_board.outcome().map(|o| o.pov(root_board.next_player().other()));
    tree.nodes.push(Node::new(None, root_outcome));

    for _ in 0..iterations {
        //we've solved the root node, so we're done
        if tree[0].solution().is_some() { break; }

        mcts_solver_step(&mut tree, 0, &root_board, exploration_weight, rng);
    }

    tree
}

pub struct MCTSBot<R: Rng> {
    iterations: u64,
    exploration_weight: f32,
    rng: R,
}

impl<R: Rng> Debug for MCTSBot<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MCTSBot {{ iterations: {}, exploration_weight: {} }}", self.iterations, self.exploration_weight)
    }
}

impl<R: Rng> MCTSBot<R> {
    pub fn new(iterations: u64, exploration_weight: f32, rng: R) -> Self {
        assert!(iterations > 0);
        MCTSBot { iterations, exploration_weight, rng }
    }
}

impl<R: Rng, B: Board> Bot<B> for MCTSBot<R> {
    fn select_move(&mut self, board: &B) -> B::Move {
        assert!(!board.is_done());

        let tree = mcts_build_tree(board, self.iterations, self.exploration_weight, &mut self.rng);
        tree.best_move()
    }
}
