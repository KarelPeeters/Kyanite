use crossbeam::channel::Sender;
use itertools::Itertools;
use rand::thread_rng;
use sttt::board::Board;
use sttt::mcts::mcts_build_tree;

use crate::selfplay::{Generator, Message, MoveSelector, Position, Simulation, StartGameCounter};

#[derive(Debug)]
pub struct MCTSGeneratorSettings {
    pub thread_count: usize,

    pub iterations: u64,
    pub exploration_weight: f32,
}

impl Generator for MCTSGeneratorSettings {
    type ThreadParam = ();

    fn thread_params(&self) -> Vec<Self::ThreadParam> {
        vec![(); self.thread_count]
    }

    fn thread_main(
        &self,
        move_selector: &MoveSelector,
        _thread_param: (),
        start_counter: &StartGameCounter,
        sender: &Sender<Message>,
    ) {
        let mut rng = thread_rng();

        loop {
            // check if we should stop
            if start_counter.request_up_to(1) == 0 {
                return;
            }

            let mut positions = Vec::new();
            let mut board = Board::new();

            let final_won_by = loop {
                match board.won_by {
                    Some(player) => {
                        break player;
                    }
                    None => {
                        let tree = mcts_build_tree(&board, self.iterations, 2.0, &mut thread_rng());
                        sender.send(Message::Counter { evals: self.iterations, moves: 1 }).unwrap();

                        let root = &tree[0];
                        let children = root.children().unwrap();

                        let policy = children.iter().map(|child| {
                            //TODO same issue as in zero::Tree, isn't this off by one?
                            (tree[child].visits as f32) / (root.visits as f32)
                        }).collect_vec();

                        let picked_index = move_selector.select(board.count_tiles(), &policy, &mut rng);
                        let picked_child = children.get(picked_index);
                        let picked_move = tree[picked_child].coord;

                        positions.push(Position {
                            board: board.clone(),
                            should_store: true,
                            eval: tree.eval(),
                            policy,
                        });

                        board.play(picked_move);
                    }
                }
            };

            let simulation = Simulation { won_by: final_won_by, positions };
            sender.send(Message::Simulation(simulation)).unwrap();
        }
    }
}