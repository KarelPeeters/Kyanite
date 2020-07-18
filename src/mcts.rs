use std::f32;

use ordered_float::OrderedFloat;
use rand::Rng;
use rand::seq::IteratorRandom;

use crate::board::{Board, Coord, Player};

#[derive(Clone)]
struct Node {
    board: Board,
    children: Option<Vec<usize>>,
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

    fn to_string(&self, tree: &Vec<Node>, depth: usize) -> String {
        let mut res = String::new();
        self.append_tree(tree, &mut res, 0, depth, true);
        res.shrink_to_fit();
        res
    }

    fn append_tree(&self, tree: &Vec<Node>, out: &mut String, depth: usize, max_depth: usize, is_last: bool) {
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
        out.push_str(&format!("{:?}: {}/{}", self.board.last_move, self.wins, self.visits));
        out.push('\n');

        if let Some(children) = &self.children {
            for (i, child) in children.iter().enumerate() {
                tree[*child].append_tree(tree, out, depth + 1, max_depth, i == children.len() - 1)
            }
        };
    }
}

pub fn old_move_mcts(board: &Board, iterations: usize, rand: &mut impl Rng, log: bool) -> Option<Coord> {
    let mut tree: Vec<Node> = Vec::with_capacity(iterations);
    tree.push(Node::new(Board::new()));

    while tree[0].visits < iterations {
        if log && tree[0].visits % (iterations / 20) == 0 {
            println!("Progress: {}", tree[0].visits as f64 / iterations as f64);
        }

        old_recurse_down(&mut tree, 0, &board, board.next_player, rand);
    }

    if log {
        println!("{}", tree[0].to_string(&tree, 1));
        println!("{:?}", count(&tree, 0));
        println!("Total node count {:?}", tree.len());
    }

    tree[0].children.as_ref().and_then(
        |children| children.iter()
            .max_by_key(|n| tree[**n].visits)
            .map(|n| tree[*n].board.last_move)
            .and_then(|x| x)
    )
}

fn count(tree: &Vec<Node>, node: usize) -> (u32, u32) {
    if tree[node].children.clone().map_or(false, |children| children.iter().all(|n| tree[*n].visits == 0)) {
        (1, 1)
    } else {
        let (c, u) = tree[node].children.iter()
            .flat_map(|children| children.iter().map(|child| count(tree, *child)))
            .fold((0, 0), |(ac, au), (c, u)| (ac + c, au + u));
        (c + 1, u)
    }
}

/*pub fn move_mcts(board: &Board, iterations: u32, rand: &mut impl Rng) -> Option<Coord> {
    let mut head = Node::new(Coord::none());
    let mut visited: Vec<usize> = Vec::with_capacity(81);

    while head.visits < iterations {
        //selection
        let mut current = &mut head;
        while let Some(children) = current.children {
            let i = children.iter().enumerate()
                .filter(|(_, n)| n.visits == 0)
                .max_by_key(|(_, n)| uct(n.wins, n.visits, current.visits))
                .map(|(i, _)| i)
                .expect("No children?");

            visited.push(i);
            current = unsafe {

            }
            current = &mut visited.last().unwrap().1[i];

            //TODO how to get reference inside Some child?
        }

        //expansion

        //simulation
        let mut won: bool = unimplemented!();

        //backpropagation
        let mut curr = &head;

        for &i in visited.iter() {
            curr.increment(won);
            won = !won;
            curr = &curr.children.unwrap()[i];
        }
        curr.increment(won);
    }

    None
}*/

fn old_recurse_down<R: Rng>(tree: &mut Vec<Node>, node: usize, board: &Board, player: Player, rand: &mut R) -> bool {
    let won = if let Some(winner) = board.won_by {
        is_win(winner, player, rand)
    } else {
        //TODO children doesn't have to be a vec, it could just be an (index, len) tuple
        let mut children = tree[node].children.take().unwrap_or_else(|| board.available_moves().map(|c| {
            let mut board = board.clone();
            board.play(c);


            let index = tree.len();
            tree.push(Node::new(board));
            index
        }).collect());
        let explore_child = children.iter_mut().filter(|n| tree[**n].visits == 0).choose(rand);

        let won = if let Some(next) = explore_child {
            let next_node = &mut tree[*next];

            //Exploration
            let mut board = next_node.board.clone();

            //Simulation
            loop {
                match board.random_available_move(rand) {
                    Some(mv) => board.play(mv),
                    None => break
                };
            }

            let won = is_win(board.won_by.unwrap(), player, rand);
            next_node.visits += 1;
            if won {
                next_node.wins += 1;
            }
            won
        } else {
            //Selection
            let next = children.iter_mut()
                .max_by_key(|n| {
                    let n = &tree[**n];
                    uct(n.wins, n.visits, tree[node].visits)
                }).unwrap();
            old_recurse_down(tree, *next, &tree[*next].board.clone(), player, rand)
        };

        tree[node].children = Some(children);
        won
    };

    tree[node].visits += 1;
    if won {
        tree[node].wins += 1;
    }
    won
}

/*
//TODO continue rewrite procedurally using https://stackoverflow.com/questions/29296038/implementing-a-mutable-tree-structure
pub fn move_mcts<R: Rng>(board: &Board, iterations: u32, rand: &mut R) -> Option<Coord> {
    let mut head = Node::new(Coord::none());

    while head.visits < iterations {
        let mut visited: Vec<usize> = Vec::with_capacity(81);

        let mut current = &mut head;
        let mut current_board = board.clone();

        while !board.is_done() {
            let mut children: Vec<Node>;
            let next_index: usize;

            match current.children.take() {
                None => {
                    children = board.available_moves().map(|c| Node::new(c)).collect();
                    next_index = rand.gen_range(0, children.len());
                }
                Some(c) => {
                    children = c;
                    next_index = children.iter().enumerate().filter(|(_, n)| n.visits == 0).choose(rand).map(|(i, _)| i)
                        .or_else(|| children.iter().enumerate().max_by_key(|(_, n)| uct(n.wins, n.visits, current.visits)).map(|(i, _)| i))
                        .unwrap();
                }
            };

            visited.push(next_index);
            let next = &mut children[next_index];

            current_board.play(next.coord);

            current = next;
            current.children = Some(children);
        }

        loop {
            match current_board.random_available_move(rand) {
                Some(mv) => current_board.play(mv),
                None => break,
            };
        }

        let won = match current_board.won_by.unwrap() {
            Player::Neutral => rand.gen(),
            player => player == board.next_player,
        };

        head.increment(won);
        let mut current = &head;
        for i in visited {
            current = &current.children.unwrap()[i];
            current.increment(won)
        }
    }

    head.children.and_then(|children| children.iter().max_by_key(|n| n.visits).map(|n| n.coord))
}
*/

fn uct(wins: usize, visits: usize, parent_visits: usize) -> OrderedFloat<f32> {
    let wins = wins as f32;
    let visits = visits as f32;
    let parent_visits = parent_visits as f32;

    let value = wins / visits + 1.5 * (parent_visits.ln() / visits).sqrt();
    OrderedFloat(value)
}

fn is_win<R: Rng>(winner: Player, player: Player, rand: &mut R) -> bool {
    (winner == player) || (winner == Player::Neutral && rand.gen())
}