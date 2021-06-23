use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam::channel::{Sender, SendError};
use itertools::Itertools;
use rand::thread_rng;
use sttt::board::Board;
use sttt::mcts::mcts_build_tree;

use crate::selfplay::{Generator, Message, MoveSelector, Position, Simulation};

#[derive(Debug)]
pub struct MCTSGeneratorSettings {
    pub thread_count: usize,

    pub iterations: u64,
    pub exploration_weight: f32,
}

impl Generator for MCTSGeneratorSettings {
    type Init = ();
    type ThreadInit = ();

    fn initialize(&self) -> Self::Init {
        ()
    }

    fn thread_params(&self) -> Vec<Self::ThreadInit> {
        vec![(); self.thread_count]
    }

    fn thread_main(
        &self,
        move_selector: &MoveSelector,
        _: &(), _: (),
        request_stop: &AtomicBool,
        sender: &Sender<Message>,
    ) -> Result<(), SendError<Message>> {
        let mut rng = thread_rng();

        loop {
            let mut positions = Vec::new();
            let mut board = Board::new();

            let final_won_by = loop {
                //early exit
                if request_stop.load(Ordering::Relaxed) { return Ok(()); }

                match board.won_by {
                    Some(player) => {
                        break player;
                    }
                    None => {
                        let tree = mcts_build_tree(&board, self.iterations, 2.0, &mut thread_rng());
                        sender.send(Message::Counter { evals: self.iterations, moves: 1 })?;

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
                            value: tree.eval().value(),
                            policy,
                        });

                        board.play(picked_move);
                    }
                }
            };

            let simulation = Simulation { won_by: final_won_by, positions };
            sender.send(Message::Simulation(simulation))?;
        }
    }
}