use crossbeam::channel::Sender;
use itertools::Itertools;
use rand::thread_rng;
use sttt::ai::mcts::mcts_build_tree;
use sttt::board::Board;

use crate::evaluation::{WDL, ZeroEvaluation};
use crate::selfplay::{Generator, Message, MoveSelector, Position, Simulation, StartGameCounter};

#[derive(Debug)]
pub struct MCTSGeneratorSettings {
    pub thread_count: usize,

    pub iterations: u64,
    pub exploration_weight: f32,
}

impl<B: Board> Generator<B> for MCTSGeneratorSettings {
    type ThreadParam = ();

    fn thread_params(&self) -> Vec<Self::ThreadParam> {
        vec![(); self.thread_count]
    }

    fn thread_main(
        &self,
        start_board: &B,
        move_selector: MoveSelector,
        _thread_param: (),
        start_counter: &StartGameCounter,
        sender: Sender<Message<B>>,
    ) {
        let mut rng = thread_rng();

        loop {
            // check if we should stop
            if start_counter.request_up_to(1) == 0 {
                return;
            }

            let mut positions = Vec::new();
            let mut board = start_board.clone();
            let mut move_count = 0;

            let outcome = loop {
                match board.outcome() {
                    Some(outcome) => {
                        break outcome;
                    }
                    None => {
                        let tree = mcts_build_tree(&board, self.iterations, self.exploration_weight, &mut thread_rng());
                        sender.send(Message::Counter { evals: self.iterations, moves: 1 }).unwrap();

                        let root = &tree[0];
                        let children = root.children.unwrap();

                        let policy = children.iter().map(|child| {
                            tree[child].visits as f32 / root.visits as f32
                        }).collect_vec();

                        let picked_index = move_selector.select(move_count, policy.iter().copied(), &mut rng);
                        let picked_child = children.get(picked_index);
                        let picked_move = tree[picked_child].last_move.unwrap();

                        let eval = tree.eval();
                        positions.push(Position {
                            board: board.clone(),
                            should_store: true,
                            evaluation: ZeroEvaluation {
                                wdl: WDL { win: eval.win, draw: eval.draw, loss: eval.loss },
                                policy,
                            },
                        });

                        board.play(picked_move);
                        move_count += 1;
                    }
                }
            };

            let simulation = Simulation { outcome, positions };
            sender.send(Message::Simulation(simulation)).unwrap();
        }
    }
}