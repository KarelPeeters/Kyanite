use std::fmt::Debug;

use board_game::board::Board;

use crate::mapping::BoardMapper;
use crate::muzero::step::{muzero_step_apply, muzero_step_gather, MuZeroExpandRequest, MuZeroRequest, MuZeroResponse};
use crate::muzero::tree::MuTree;
use crate::network::muzero::MuZeroFusedExecutors;
use crate::zero::node::UctWeights;
use crate::zero::step::FpuMode;

#[derive(Debug, Copy, Clone)]
pub struct MuZeroSettings {
    pub batch_size: usize,
    pub weights: UctWeights,
    pub use_value: bool,
    pub fpu_mode: FpuMode,
    pub top_moves: usize,
}

impl MuZeroSettings {
    pub fn new(batch_size: usize, weights: UctWeights, use_value: bool, fpu_mode: FpuMode, top_moves: usize) -> Self {
        Self {
            batch_size,
            weights,
            use_value,
            fpu_mode,
            top_moves,
        }
    }
}

impl MuZeroSettings {
    /// Construct a new tree from scratch on the given board.
    pub fn build_tree<B: Board, M: BoardMapper<B>>(
        self,
        root_board: &B,
        networks: &mut MuZeroFusedExecutors<B, M>,
        stop: impl FnMut(&MuTree<B>) -> bool,
    ) -> MuTree<B> {
        let mut tree = MuTree::new(root_board.clone(), networks.mapper.policy_len());
        self.expand_tree(&mut tree, networks, stop);
        tree
    }

    // Continue expanding an existing tree.
    pub fn expand_tree<B: Board, M: BoardMapper<B>>(
        self,
        tree: &mut MuTree<B>,
        network: &mut MuZeroFusedExecutors<B, M>,
        mut stop: impl FnMut(&MuTree<B>) -> bool,
    ) {
        assert_eq!(network.root_exec.batch_size, 1);
        assert_eq!(network.expand_exec.batch_size, 1);

        'outer: loop {
            if stop(tree) {
                break 'outer;
            }

            // gather next request
            let request = muzero_step_gather(tree, self.weights, self.use_value, self.fpu_mode);

            // evaluate request
            if let Some(request) = request {
                let response = match request {
                    MuZeroRequest::Root { node, board } => {
                        let (state, eval) = network.eval_root(&[board]).remove(0);
                        MuZeroResponse { node, state, eval }
                    }
                    MuZeroRequest::Expand(MuZeroExpandRequest {
                        node,
                        state,
                        move_index,
                    }) => {
                        let (state, eval) = network.eval_expand(&[(state, move_index)]).remove(0);
                        MuZeroResponse { node, state, eval }
                    }
                };

                // apply response
                muzero_step_apply(tree, self.top_moves, response, network.mapper);
            };
        }
    }
}
