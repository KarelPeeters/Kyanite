use derive_more::Constructor;
use ordered_float::OrderedFloat;
use rand::Rng;

use crate::board::{Board, Coord, Player};
use crate::bot_game::Bot;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct IdxRange {
    start: usize,
    end: usize,
}

impl IdxRange {
    fn iter(self) -> std::ops::Range<usize> {
        self.start..self.end
    }
}

impl IntoIterator for IdxRange {
    type Item = usize;
    type IntoIter = std::ops::Range<usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

struct Node {
    coord: Coord,
    children: Option<IdxRange>,
    visits: usize,
    wins: usize,
}

impl Node {
    fn new(coord: Coord) -> Self {
        Node {
            coord,
            children: None,
            visits: 0,
            wins: 0,
        }
    }

    fn uct(&self, parent_visits: usize) -> OrderedFloat<f32> {
        let wins = self.wins as f32;
        let visits = self.visits as f32;
        let value = (wins / visits) + (2.0 * (parent_visits as f32).ln() / visits).sqrt();
        value.into()
    }
}

#[derive(Constructor)]
pub struct MCTSBot<R: Rng> {
    iterations: usize,
    rand: R,
}

impl<R: Rng> Bot for MCTSBot<R> {
    fn play(&mut self, board: &Board) -> Option<Coord> {
        let mut tree: Vec<Node> = Vec::new();
        let mut visited: Vec<usize> = Vec::with_capacity(81);

        //the actual coord doesn't matter, just pick something
        tree.push(Node::new(Coord::from_o(0)));

        for _ in 0..self.iterations {
            //println!("Start iter {}", i);
            let mut curr_node = 0;
            let mut curr_board = board.clone();

            visited.clear();
            visited.push(curr_node);

            while !curr_board.is_done() {
                //Init children
                let children = match tree[curr_node].children {
                    Some(children) => children,
                    None => {
                        //println!("Init {}", tree[curr_node].coord.o());

                        let start = tree.len();
                        tree.extend(curr_board.available_moves().map(Node::new));
                        let end = tree.len();

                        let children = IdxRange { start, end };
                        tree[curr_node].children = Some(children);
                        children
                    }
                };

                //Exploration
                let unexplored_children = children.iter()
                    .filter(|&c| tree[c].visits == 0);
                let count = unexplored_children.clone().count();

                if count != 0 {
                    let child = unexplored_children.clone().nth(self.rand.gen_range(0, count))
                        .expect("we specifically selected the index based on the count already");

                    curr_node = child;
                    visited.push(curr_node);
                    curr_board.play(tree[curr_node].coord);

                    //println!("Exploring {}", tree[curr_node].coord.o());

                    break;
                }

                //Selection
                let parent_visits = tree[curr_node].visits;
                //println!("Values: {:?}", children.iter().map(|child| tree[child].uct(parent_visits)).collect_vec());

                let selected = children.iter().max_by_key(|&child| {
                    tree[child].uct(parent_visits)
                }).expect("Board is not done, this node should have a child");

                curr_node = selected;
                curr_board.play(tree[curr_node].coord);
                visited.push(curr_node);

                //println!("Selecting {}", tree[curr_node].coord.o());
            }

            //Simulate
            let won_by = loop {
                if let Some(won_by) = curr_board.won_by {
                    break won_by;
                }

                curr_board.play(curr_board.random_available_move(&mut self.rand)
                    .expect("No winner, so board is not done yet"));
            };

            //println!("Simulation won by {:?}", won_by);

            //Update
            let mut won = if won_by != Player::Neutral {
                won_by == board.next_player
            } else {
                self.rand.gen()
            };

            for &node in &visited {
                won ^= true;
                let node = &mut tree[node];
                node.visits += 1;
                if won {
                    node.wins += 1;
                }
            }
        }

        match tree[0].children {
            None => board.random_available_move(&mut self.rand),
            Some(children) => {
                children.iter().rev().max_by_key(|&child| {
                    tree[child].visits
                }).map(|child| {
                    tree[child].coord
                })
            }
        }
    }
}
