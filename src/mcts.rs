use std::fmt::Write;
use std::num::NonZeroUsize;
use std::ops::Range;

use ordered_float::OrderedFloat;
use rand::Rng;

use crate::board::{Board, Coord, Player};
use crate::bot_game::Bot;

#[derive(Copy, Clone)]
struct TreeRange {
    start: NonZeroUsize,
    end: usize,
}

impl IntoIterator for TreeRange {
    type Item = usize;
    type IntoIter = Range<usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.start.get()..self.end
    }
}

struct Node {
    board: Board,
    children: Option<TreeRange>,

    //score from the perspective of the next player at this node
    // +1: win, 0: draw, -1: loss
    //TODO be careful when implementing UCT!
    score: i32,
    visits: i32,
}

struct State {
    exploration_factor: f32,
    nodes: Vec<Node>,
}

impl State {
    // returns the achieved score from the perspective of the next player at the node
    fn recurse_down<R: Rng>(&mut self, rand: &mut R, node: usize) -> (i32, i32) {
        //if this is a terminal node, juts return the score immediately
        if let Some(winner) = self.nodes[node].board.won_by {
            //TODO this should always be either 1 or -1, figure out which and return that
            return (winner_to_score(winner, self.nodes[node].board.next_player), 1);
        }

        let mut total_score: i32 = 0;
        let mut total_visits: i32 = 0;

        //get the children, initialize if necessary
        let children = match self.nodes[node].children {
            Some(children) => children,
            None => {
                let start = self.nodes.len();
                let board = self.nodes[node].board.clone();

                //initialize all child nodes
                for mv in board.available_moves() {
                    let mut next_board = board.clone();
                    next_board.play(mv);

                    //do the first random playout on this board immediately
                    let winner = random_playout(rand, next_board.clone());
                    let next_score = winner_to_score(winner, next_board.next_player);

                    total_score += next_score;
                    total_visits += 1;

                    let next_node = Node {
                        score: next_score,
                        visits: 1,
                        children: None,
                        board: next_board,
                    };
                    self.nodes.push(next_node);
                }

                let end = self.nodes.len();

                let children = TreeRange { start: NonZeroUsize::new(start).unwrap(), end };
                self.nodes[node].children = Some(children);
                children
            }
        };

        //find the best child
        //there is guaranteed to be at least one child since this is not a terminal node
        let best_child = children.into_iter().max_by_key(|&child_node| {
            let child_node = &self.nodes[child_node];
            self.uct(child_node.score, child_node.visits, self.nodes[node].visits)
        }).unwrap();

        //evaluate that child
        let (child_score, child_visits) = self.recurse_down(rand, best_child);

        //flip the score because the perspective changes
        total_score += -child_score;
        total_visits += child_visits;

        //Actually increment the current node
        self.nodes[node].score += total_score;
        self.nodes[node].visits += total_visits;

        (total_score, total_visits)
    }

    fn uct(&self, score: i32, visits: i32, parent_visits: i32) -> OrderedFloat<f32> {
        let score = score as f32;
        let visits = visits as f32;
        let parent_visits = parent_visits as f32;

        let exploitation = (score + visits) / (2.0 * visits);
        let exploration = (parent_visits.ln() / visits).sqrt();

        (exploitation + self.exploration_factor * exploration).into()
    }

    fn tree_to_string(&self, node: usize, depth: usize) -> String {
        fn print_tree_rec(state: &State, result: &mut String, mv: Option<Coord>, node: usize, max_depth: usize, curr_depth: usize, is_last: bool) {
            if curr_depth == max_depth { return; }

            for _ in 1..curr_depth {
                result.push_str("│ ");
            }

            if curr_depth != 0 {
                if is_last {
                    result.push_str("└─")
                } else {
                    result.push_str("├─")
                }
            }

            let node = &state.nodes[node];
            let children = node.children;

            if children.is_some() {
                result.push_str("┬")
            } else {
                result.push_str("─")
            }

            write!(result, " {:?} {}/{} = {}\n", mv, node.score, node.visits, node.score as f32 / node.visits as f32).unwrap();

            if let Some(children) = children {
                for (child_mv, child) in node.board.available_moves().zip(children.into_iter()) {
                    print_tree_rec(state, result, Some(child_mv), child, max_depth, curr_depth + 1, child == children.end - 1)
                }
            }
        }

        let mut string = String::new();
        print_tree_rec(self, &mut string, None, node, depth, 0, false);
        return string;
    }
}

fn random_playout<R: Rng>(rand: &mut R, mut board: Board) -> Player {
    loop {
        match board.random_available_move(rand) {
            None => return board.won_by.unwrap(),
            Some(mv) => board.play(mv),
        };
    }
}

fn winner_to_score(winner: Player, perspective: Player) -> i32 {
    debug_assert!(perspective == Player::X || perspective == Player::O);

    if winner == Player::Neutral { 0 } else if winner == perspective { 1 } else { -1 }
}

pub struct MCTSBot<R: Rng> {
    iterations: usize,
    exploration_factor: f32,
    rand: R,
}

impl<R: Rng> MCTSBot<R> {
    pub fn new(iterations: usize, rand: R) -> Self {
        MCTSBot {
            iterations,
            exploration_factor: 1.5,
            rand,
        }
    }
}

impl<R: Rng> Bot for MCTSBot<R> {
    fn play(&mut self, board: &Board) -> Option<Coord> {
        //initialize state
        let root = Node {
            board: board.clone(),
            children: None,
            score: 0,
            visits: 0,
        };

        let mut state = State {
            exploration_factor: self.exploration_factor,
            nodes: vec![root],
        };

        //actually run
        for i in 0..self.iterations {
            if i % (self.iterations / 10) == 0 {
                // println!("Progress: {}/{}", i, self.iterations);
            }

            state.recurse_down(&mut self.rand, 0);
        }

        // println!("{}", state.tree_to_string(0, 2));

        //recover the best move
        let children = state.nodes[0].children?;

        let best_child = children.into_iter().max_by_key(|&child| {
            state.nodes[child].visits
        })?;

        let best_move = state.nodes[0].board.available_moves()
            .nth(best_child - children.start.get())
            .unwrap();

        Some(best_move)
    }
}