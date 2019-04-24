//TODO
// * smallvec for children
// * merge nodes, add Vec for parents
// * allocate 81-moveCount instead of 81
// * downsize Vec after children are added

use std::collections::HashMap;

use ordered_float::OrderedFloat;
use rand::Rng;

use crate::board::{Board, Coord, Player};
use rand::seq::SliceRandom;

type Hash = usize;

struct Node {
    board: Board,
    children: Option<Vec<Hash>>,

    visits: usize,
    wins: usize,
}

impl Node {
    fn new(board: Board) -> Node {
        Node { board, children: None, visits: 0, wins: 0 }
    }

    fn increment(&mut self, won: bool) {
        self.visits += 1;
        self.wins += won as usize;
    }

    fn hash(&self) -> Hash {
        self.board.get_hash()
    }

    fn uct(&self, parent_visits: usize) -> OrderedFloat<f32> {
        let wins = self.wins as f32;
        let visits = self.visits as f32;
        let parent_visits = parent_visits as f32;

        let value = wins / visits + 1.5 * (parent_visits.ln() / visits).sqrt();
        OrderedFloat(value)
    }
}

pub fn hmcts_move(board: &Board, iterations: usize, rng: &mut impl Rng) -> Option<Coord> {
    let mut nodes: HashMap<Hash, Node> = HashMap::new();

    let root = Node::new(board.clone());
    let root_hash = root.hash();
    nodes.insert(root_hash, root);

    for _ in 0..iterations {
        let mut visited: Vec<Hash> = Vec::with_capacity(81);
        visited.push(root_hash);

        //Selection
        let mut current = nodes.get(&root_hash).unwrap();
        while let Some(children) = &current.children {
            let (h, n) = children.iter()
                .map(|h| (h, nodes.get(h).unwrap()))
                .max_by_key(|(_, n)| n.uct(current.visits))
                .unwrap();

            current = n;
            visited.push(*h);
        }

        let won_by: Player = match board.won_by {
            Some(player) => player,
            None => {
                //Expansion
                let current_hash = current.hash();
                let mut children: Vec<Node> = Vec::new();

                //create children
                for mv in current.board.available_moves() {
                    let mut board = current.board.clone();
                    board.play(mv);
                    let node = Node::new(board);
                    children.push(node);
                }

                //pick random child
                let picked = children.choose(rng).unwrap();
                let picked_hash = picked.hash();
                visited.push(picked_hash);

                let mut child_hashes = Vec::with_capacity(children.len());
                for child in children {
                    child_hashes.push(child.hash());
                    nodes.insert(child.hash(), child);
                }

                nodes.get_mut(&current_hash).unwrap().children = Some(child_hashes);

                //simulation
                let mut board = nodes.get(&picked_hash).unwrap().board.clone();
                while let Some(mv) = board.random_available_move(rng) {
                    board.play(mv);
                }

                board.won_by.unwrap()
            },
        };

        //Backpropagation
        //if won_by is a player it must be the last player
        let mut won = won_by != Player::Neutral || rng.gen();
        for h in visited.iter().rev() {
            nodes.get_mut(h).unwrap().increment(won);
            won = !won;
        }
    }

    let root = nodes.get(&root_hash).unwrap();
    let best = root.children.iter().flat_map(|c| c.iter().flat_map(|h| nodes.get(h))).max_by_key(|n| n.uct(root.visits));

    best.and_then(|n| n.board.last_move)
}