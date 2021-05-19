use std::num::NonZeroUsize;
use std::ops::{Index, IndexMut};

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
    pub fn iter(&self) -> std::ops::Range<usize> {
        self.start.get()..(self.start.get() + self.length as usize)
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
pub struct Node {
    pub coord: Coord,
    //this is not just a Option<IdxRange> because of struct layout inefficiencies
    children_start: usize,
    children_length: u8,

    pub p: f32,
    pub n: u64,
    pub total_q: f32,
}

impl Node {
    fn new(coord: Coord, p: f32) -> Self {
        Node {
            coord,
            children_start: 0,
            children_length: 0,

            p,
            n: 0,
            total_q: 0.0,
        }
    }

    pub fn uct(&self) -> f32 {
        //TODO don't they use something completely different in the AlphhaGo Zero paper?
        
        let q = self.total_q / self.n as f32;
        let u = self.p / (1 + self.n) as f32;
        q + u
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

    pub fn best_move(&self) -> Coord {
        let children = self[0].children()
            .expect("Root node must have children");

        let best_child = children.iter().rev().max_by_key(|&child| {
            self[child].n
        }).expect("Root node must have non-empty children");

        self[best_child].coord
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

pub fn mcts_zero_build_tree(board: &Board, iterations: u64, network: &mut Network) -> Tree {
    assert!(iterations > 0, "MCTS must run for at least 1 iteration");
    assert!(!board.is_done(), "Cannot build MCTS tree for done board");

    let mut tree = Tree::new(board.clone());
    let mut parent_list = Vec::with_capacity(81);

    //the actual coord doesn't matter, just pick something
    tree.nodes.push(Node::new(Coord::from_o(0), 1.0));

    for _ in 0..iterations {
        parent_list.clear();

        let mut curr_node: usize = 0;
        let mut curr_board = board.clone();

        let mut value = loop {
            parent_list.push(curr_node);

            // if the game is done return the actual value
            if let Some(won_by) = curr_board.won_by {
                // TODO should this be 1 or -1?
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

                    break evaluation.value;
                }
                Some(children) => children,
            };

            //continue with the best child
            let selected = children.iter().max_by_key(|&child| {
                OrderedFloat(tree[child].uct())
            }).expect("Board is not done, this node should have a child");

            curr_node = selected;
            curr_board.play(tree[curr_node].coord);
        };

        for &update_node in parent_list.iter().rev() {
            value = -value;

            let node = &mut tree[update_node];
            node.n += 1;
            node.total_q += value;
        }
    }

    assert_eq!(iterations, tree[0].n, "implementation error");
    tree
}

pub struct MCTSZeroBot {
    iterations: u64,
    network: Network,
}

impl MCTSZeroBot {
    pub fn new(iterations: u64, network: Network) -> Self {
        MCTSZeroBot { iterations, network }
    }
}

impl Bot for MCTSZeroBot {
    fn play(&mut self, board: &Board) -> Option<Coord> {
        if board.is_done() {
            None
        } else {
            let tree = mcts_zero_build_tree(board, self.iterations, &mut self.network);
            Some(tree.best_move())
        }
    }
}