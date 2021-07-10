use std::num::NonZeroUsize;
use std::ops::{Index, IndexMut};

use decorum::N32;
use itertools::Itertools;
use rand::Rng;

use crate::ai::Bot;
use crate::board::{Board, Outcome};

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

#[derive(Debug)]
pub struct Node<M> {
    pub last_move: Option<M>,
    pub children: Option<IdxRange>,
    pub wins: u64,
    pub draws: u64,
    pub visits: u64,
}

#[derive(Debug, Copy, Clone)]
pub struct Evaluation {
    pub win: f32,
    pub draw: f32,
    pub loss: f32,
}

impl std::ops::Neg for Evaluation {
    type Output = Evaluation;

    fn neg(self) -> Self::Output {
        Evaluation { win: self.loss, draw: self.draw, loss: self.win }
    }
}

impl Evaluation {
    pub fn value(&self) -> f32 {
        self.win - self.loss
    }
}

impl<M> Node<M> {
    fn new(last_move: Option<M>) -> Self {
        Node {
            last_move,
            children: None,
            wins: 0,
            draws: 0,
            visits: 0,
        }
    }

    pub fn uct(&self, parent_visits: u64, exploration_weight: f32) -> f32 {
        let visits = self.visits as f32;
        let value_unit = (self.eval().value() + 1.0) / 2.0;
        let explore = ((parent_visits as f32).ln() / visits).sqrt();

        value_unit + exploration_weight * explore
    }

    /// The value of this node from the POV of the player that could play this move.
    pub fn eval(&self) -> Evaluation {
        let visits = self.visits as f32;
        Evaluation {
            win: self.wins as f32 / visits,
            draw: self.draws as f32 / visits,
            loss: (self.visits - self.wins - self.draws) as f32 / visits,
        }
    }
}

/// A small wrapper type for Vec<Node> that uses u64 for indexing instead.
#[derive(Debug)]
pub struct Tree<B: Board> {
    pub root_board: B,
    pub nodes: Vec<Node<B::Move>>,
}

impl<B: Board> Tree<B> {
    pub fn new(root_board: B) -> Self {
        Tree { root_board, nodes: Default::default() }
    }

    pub fn best_move(&self) -> B::Move {
        let children = self[0].children
            .expect("Root node must have children");

        let best_child = children.iter().rev().max_by_key(|&child| {
            self[child].visits
        }).expect("Root node must have non-empty children");

        self[best_child].last_move.unwrap()
    }

    /// The value of `root_board` from the POV of `root_board.next_player`.
    pub fn eval(&self) -> Evaluation {
        // the evaluation of the starting board is the opposite of the the evaluation of the root node,
        //   since that is the evaluation from the POV of the "previous" player
        -self[0].eval()
    }

    pub fn print(&self, depth: u64) {
        println!("move: visits, value <- W,D,L");
        self.print_impl(0, 0, depth);
    }

    fn print_impl(&self, node: usize, depth: u64, max_depth: u64) {
        let node = &self[node];

        for _ in 0..depth { print!("  ") }
        let eval = node.eval();
        println!("{:?}: {}, {:.3} <- {:.3},{:.3},{:.3}", node.last_move, node.visits, eval.value(), eval.win, eval.draw, eval.loss);

        if depth == max_depth { return; }

        if let Some(children) = node.children {
            let values = children.iter().map(|c| self[c].eval().value());
            if values.clone().any(|v| v.is_nan()) { return; }

            let best_child_index = values.map(N32::from).position_max().unwrap();
            let best_child = children.start.get() + best_child_index;

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

pub fn mcts_build_tree<B: Board>(board: &B, iterations: u64, exploration_weight: f32, rand: &mut impl Rng) -> Tree<B> {
    assert!(iterations > 0, "MCTS must run for at least 1 iteration");
    assert!(!board.is_done(), "Cannot build MCTS tree for done board");

    let mut tree = Tree::new(board.clone());
    let mut parent_list = Vec::with_capacity(81);

    tree.nodes.push(Node::new(None));

    for _ in 0..iterations {
        parent_list.clear();

        let mut curr_node: usize = 0;
        let mut curr_board = board.clone();

        while !curr_board.is_done() {
            parent_list.push(curr_node);

            //Init children
            let children = match tree[curr_node].children {
                Some(children) => children,
                None => {
                    let start = tree.nodes.len();
                    tree.nodes.extend(
                        curr_board.available_moves()
                            .map(|c| Node::new(Some(c)))
                    );
                    let length = tree.nodes.len() - start;

                    let children = IdxRange {
                        start: NonZeroUsize::new(start).unwrap(),
                        length,
                    };
                    tree[curr_node].children = Some(children);
                    children
                }
            };

            //Exploration
            let unexplored_children = children.iter()
                .filter(|&c| tree[c].visits == 0);
            let count = unexplored_children.clone().count();

            if count != 0 {
                let child = unexplored_children.clone().nth(rand.gen_range(0..count))
                    .expect("we specifically selected the index based on the count already");

                curr_node = child;
                curr_board.play(tree[curr_node].last_move.unwrap());

                break;
            }

            //Selection
            let parent_visits = tree[curr_node].visits;

            let selected = children.iter().max_by_key(|&child| {
                let uct = tree[child].uct(parent_visits, exploration_weight);
                N32::from(uct)
            }).expect("Board is not done, this node should have a child");

            curr_node = selected;
            curr_board.play(tree[curr_node].last_move.unwrap());
        }

        //Simulate
        let curr_player = curr_board.next_player();

        let outcome = loop {
            if let Some(outcome) = curr_board.outcome() {
                break outcome;
            }

            curr_board.play(curr_board.random_available_move(rand));
        };

        parent_list.push(curr_node);

        //Update
        let mut won = outcome == Outcome::WonBy(curr_player);
        let draw = outcome == Outcome::Draw;

        for &update_node in parent_list.iter().rev() {
            won = !won && !draw;

            let node = &mut tree[update_node];
            node.visits += 1;
            node.wins += won as u64;
            node.draws += draw as u64;
        }
    }

    assert_eq!(iterations, tree[0].visits, "implementation error");
    tree
}

pub struct MCTSBot<R: Rng> {
    iterations: u64,
    exploration_weight: f32,
    rng: R,
}

impl<R: Rng> MCTSBot<R> {
    pub fn new(iterations: u64, exploration_weight: f32, rand: R) -> Self {
        MCTSBot { iterations, exploration_weight, rng: rand }
    }
}

impl<R: Rng, B: Board> Bot<B> for MCTSBot<R> {
    fn select_move(&mut self, board: &B) -> B::Move {
        mcts_build_tree(board, self.iterations, self.exploration_weight, &mut self.rng)
            .best_move()
    }
}
