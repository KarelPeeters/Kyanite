use std::num::NonZeroU64;
use std::ops::{Generator, Index, IndexMut};

use ordered_float::OrderedFloat;
use rand::{Rng, thread_rng};
use sttt::board::{Board, Coord, Player};

#[derive(Debug, Copy, Clone)]
pub struct IdxRange {
    pub start: NonZeroU64,
    pub length: u8,
}

impl IdxRange {
    pub fn iter(&self) -> std::ops::Range<u64> {
        self.start.get()..(self.start.get() + self.length as u64)
    }
}

impl IntoIterator for IdxRange {
    type Item = u64;
    type IntoIter = std::ops::Range<u64>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug)]
pub struct Node {
    pub coord: Coord,
    //this is not just a Option<IdxRange> because of struct layout inefficiencies
    children_start: u64,
    children_length: u8,
    pub wins: u64,
    pub draws: u64,
    pub visits: u64,
}

impl Node {
    fn new(coord: Coord) -> Self {
        Node {
            coord,
            children_start: 0,
            children_length: 0,
            wins: 0,
            draws: 0,
            visits: 0,
        }
    }

    pub fn uct(&self, parent_visits: u64) -> f32 {
        let wins = self.wins as f32;
        let draws = self.draws as f32;
        let visits = self.visits as f32;

        //TODO is this really the best heuristic formula? maybe let the heuristic decide the weight as well?
        (wins + 0.5 * draws) / visits +
            (2.0 * (parent_visits as f32).ln() / visits).sqrt()
    }

    /// The estimated value of this node in the range -1..1
    pub fn signed_value(&self) -> f32 {
        (2.0 * (self.wins as f32) + (self.draws as f32)) / (self.visits as f32) - 1.0
    }

    pub fn children(&self) -> Option<IdxRange> {
        NonZeroU64::new(self.children_start)
            .map(|start| IdxRange { start, length: self.children_length })
    }

    pub fn set_children(&mut self, children: IdxRange) {
        self.children_start = children.start.get();
        self.children_length = children.length;
    }
}

#[derive(Debug)]
pub struct Evaluation {
    pub best_move: Option<Coord>,
    pub value: f32,
}

/// A small wrapper type for Vec<Node> that uses u64 for indexing instead.
#[derive(Debug, Default)]
pub struct Tree(pub Vec<Node>);

impl Tree {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
}

impl Index<u64> for Tree {
    type Output = Node;

    fn index(&self, index: u64) -> &Self::Output {
        &self.0[index as usize]
    }
}

impl IndexMut<u64> for Tree {
    fn index_mut(&mut self, index: u64) -> &mut Self::Output {
        &mut self.0[index as usize]
    }
}

#[derive(Debug)]
pub struct BoardEval {

}

// pub type MCTSGen = impl Generator<Option<BoardEval>, Yield=Board, Return=Tree>;

pub fn mcts_build_tree_gen(board: &Board, iterations: u64) -> impl Generator<Option<BoardEval>, Yield=Board, Return=Tree> {
    let board = board.clone();

    move |first_value: Option<BoardEval>| {
        assert!(first_value.is_none(), "Unexpected evaluation");
        let mut rand = thread_rng();

        let mut tree = Tree::default();
        let mut parent_list = Vec::with_capacity(81);

        //the actual coord doesn't matter, just pick something
        tree.0.push(Node::new(Coord::from_o(0)));

        for _ in 0..iterations {
            let mut curr_node: u64 = 0;
            let mut curr_board = board.clone();

            while !curr_board.is_done() {
                parent_list.clear();
                parent_list.push(curr_node);

                //Init children
                let children = match tree[curr_node].children() {
                    Some(children) => children,
                    None => {
                        static_assertions::const_assert!(Board::MAX_AVAILABLE_MOVES <= u8::MAX as u32);

                        let eval: Option<BoardEval> = yield curr_board.clone();
                        println!("eval: {:?}", eval);

                        let start = tree.len();
                        tree.0.extend(curr_board.available_moves().map(|c| Node::new(c)));
                        let length = (tree.len() - start) as u8;

                        let children = IdxRange {
                            start: NonZeroU64::new(start as u64).unwrap(),
                            length,
                        };
                        tree[curr_node].set_children(children);
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
                    curr_board.play(tree[curr_node].coord);

                    break;
                }

                //Selection
                let parent_visits = tree[curr_node].visits;

                let selected = children.iter().max_by_key(|&child| {
                    let uct = tree[child].uct(parent_visits);
                    OrderedFloat(uct)
                }).expect("Board is not done, this node should have a child");

                curr_node = selected;
                curr_board.play(tree[curr_node].coord);
            }

            //Simulate
            let curr_player = curr_board.next_player;

            let won_by = loop {
                if let Some(won_by) = curr_board.won_by {
                    break won_by;
                }

                curr_board.play(curr_board.random_available_move(&mut rand)
                    .expect("No winner, so board is not done yet"));
            };

            parent_list.push(curr_node);

            //Update
            let mut won = if won_by != Player::Neutral {
                won_by == curr_player
            } else {
                rand.gen()
            };

            for &update_node in parent_list.iter().rev() {
                won = !won;

                let node = &mut tree[update_node];
                node.visits += 1;
                node.wins += won as u64;
            }
        }

        tree
    }
}
