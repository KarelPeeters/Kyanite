use std::f32;

use ordered_float::OrderedFloat;
use rand::Rng;
use rand::seq::IteratorRandom;

use crate::board::{Board, Coord, Player};

struct Node {
    coord: Coord,
    children: Option<Vec<Node>>,
    visits: u32,
    wins: u32,
}

impl Node {
    fn new(coord: Coord) -> Node {
        Node { coord, children: None, visits: 0, wins: 0 }
    }

    fn to_string(&self, depth: usize) -> String {
        let mut res = String::new();
        self.append_tree(&mut res, 0, depth, true);
        res.shrink_to_fit();
        res
    }

    fn append_tree(&self, out: &mut String, depth: usize, max_depth: usize, is_last: bool) {
        if depth > max_depth {
            return;
        }

        if depth > 0 {
            out.push_str(&format!("{: ^1$}", "", (depth - 1) * 2));
            out.push_str(match is_last {
                true => "└── ",
                false => "├── ",
            });
        }
        out.push_str(&format!("{}: {}/{}", self.coord.o(), self.wins, self.visits));
        out.push('\n');

        if let Some(children) = &self.children {
            for (i, child) in children.iter().enumerate() {
                child.append_tree(out, depth + 1, max_depth, i == children.len() - 1)
            }
        };
    }
}

pub fn move_mcts<R: Rng>(board: &Board, iterations: u32, rand: &mut R) -> Option<Coord> {
    let mut head = Node::new(Coord::none());

    for i in 0..iterations {
        /*if (i % (iterations / 100)) == 0 {
            println!("{}", i);
        }*/

        recurse_down(&mut head, board.clone(), board.next_player, rand);
    }

//    println!("{}", head.to_string(1));

    head.children.and_then(|children| children.iter().max_by_key(|n| n.visits).map(|n| n.coord))
}

//TODO rewrite procedurally using https://stackoverflow.com/questions/29296038/implementing-a-mutable-tree-structure
fn recurse_down<R: Rng>(node: &mut Node, mut board: Board, player: Player, rand: &mut R) -> bool {
    let won = if let Some(winner) = board.won_by {
        is_win(winner, player, rand)
    } else {
        let mut children = node.children.take().unwrap_or_else(|| board.available_moves().map(|c| Node::new(c)).collect());

        let explore_child = children.iter_mut().filter(|n| n.visits == 0).choose(rand);
        let won = if let Some(next) = explore_child {
            //Exploration
            board.play(next.coord);

            //Simulation
            loop {
                match board.random_available_move(rand) {
                    Some(mv) => board.play(mv),
                    None => break
                };
            }

            let won = is_win(board.won_by.unwrap(), player, rand);
            next.visits += 1;
            if won {
                next.wins += 1;
            }
            won
        } else {
            //Selection
            let next = children.iter_mut().max_by_key(|n| { OrderedFloat(uct(n.wins, n.visits, node.visits)) }).unwrap();
            board.play(next.coord);
            recurse_down(next, board, player, rand)
        };

        node.children = Some(children);
        won
    };

    node.visits += 1;
    if won {
        node.wins += 1;
    }
    won
}

fn uct(wins: u32, visits: u32, parent_visits: u32) -> f32 {
    let wins = wins as f32;
    let visits = visits as f32;
    let parent_visits = parent_visits as f32;
    wins / visits + 1.5 * (parent_visits.ln() / visits).sqrt()
}

fn is_win<R: Rng>(winner: Player, player: Player, rand: &mut R) -> bool {
    (winner == player) || (winner == Player::Neutral && rand.gen())
}