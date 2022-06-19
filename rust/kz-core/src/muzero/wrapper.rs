use board_game::ai::Bot;
use std::fmt::{Debug, Formatter};

use board_game::board::AltBoard;
use cuda_nn_eval::quant::QuantizedStorage;

use crate::mapping::BoardMapper;
use crate::muzero::step::{
    muzero_step_apply, muzero_step_gather, MuZeroExpandRequest, MuZeroRequest, MuZeroResponse, MuZeroRootRequest,
};
use crate::muzero::tree::MuTree;
use crate::network::muzero::{ExpandArgs, MuZeroExpandExecutor, MuZeroRootExecutor, RootArgs};
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
    pub fn build_tree<B: AltBoard, M: BoardMapper<B>>(
        self,
        root_board: &B,
        draw_depth: u32,
        root_exec: &mut MuZeroRootExecutor<B, M>,
        expand_exec: &mut MuZeroExpandExecutor<B, M>,
        stop: impl FnMut(&MuTree<B, M>) -> bool,
    ) -> MuTree<B, M> {
        assert_eq!(root_exec.mapper, expand_exec.mapper);

        let mut tree = MuTree::new(root_board.clone(), draw_depth, root_exec.mapper);
        self.expand_tree(&mut tree, root_exec, expand_exec, stop);
        tree
    }

    // Continue expanding an existing tree.
    // TODO maybe unify this code with the async implementation written for the generator?
    pub fn expand_tree<B: AltBoard, M: BoardMapper<B>>(
        self,
        tree: &mut MuTree<B, M>,
        root_exec: &mut MuZeroRootExecutor<B, M>,
        expand_exec: &mut MuZeroExpandExecutor<B, M>,
        mut stop: impl FnMut(&MuTree<B, M>) -> bool,
    ) {
        let device = root_exec.root_exec.handles.device();
        assert_eq!(device, expand_exec.expand_exec.handles.device());
        assert_eq!(root_exec.info, expand_exec.info);

        let state_size = root_exec.info.state_saved_shape(tree.mapper).size().eval(1);

        'outer: loop {
            if stop(tree) {
                break 'outer;
            }

            // gather next request
            let request = muzero_step_gather(tree, self.weights, self.use_value, self.fpu_mode);

            // evaluate request
            if let Some(request) = request {
                let output_state = QuantizedStorage::new(device.alloc(state_size), state_size);

                let response = match request {
                    MuZeroRequest::Root(MuZeroRootRequest { node, board }) => {
                        let root_args = RootArgs {
                            board,
                            output_state: output_state.clone(),
                        };
                        let eval = root_exec.eval_root(&[root_args]).remove(0);
                        MuZeroResponse {
                            node,
                            state: output_state,
                            eval,
                        }
                    }
                    MuZeroRequest::Expand(MuZeroExpandRequest {
                        node,
                        state,
                        move_index,
                    }) => {
                        let expand_args = ExpandArgs {
                            state,
                            move_index,
                            output_state: output_state.clone(),
                        };
                        let eval = expand_exec.eval_expand(&[expand_args]).remove(0);
                        MuZeroResponse {
                            node,
                            state: output_state,
                            eval,
                        }
                    }
                };

                // apply response
                muzero_step_apply(tree, self.top_moves, response);
            };
        }
    }
}

pub struct MuZeroBot<B: AltBoard, M: BoardMapper<B>> {
    settings: MuZeroSettings,
    visits: u64,
    mapper: M,
    root_exec: MuZeroRootExecutor<B, M>,
    expand_exec: MuZeroExpandExecutor<B, M>,
}

impl<B: AltBoard, M: BoardMapper<B>> Debug for MuZeroBot<B, M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MuZeroBot")
            .field("settings", &self.settings)
            .field("visits", &self.visits)
            .field("mapper", &self.mapper)
            .finish()
    }
}

impl<B: AltBoard, M: BoardMapper<B>> MuZeroBot<B, M> {
    pub fn new(
        settings: MuZeroSettings,
        visits: u64,
        mapper: M,
        root_exec: MuZeroRootExecutor<B, M>,
        expand_exec: MuZeroExpandExecutor<B, M>,
    ) -> Self {
        Self {
            settings,
            visits,
            mapper,
            root_exec,
            expand_exec,
        }
    }
}

impl<B: AltBoard, M: BoardMapper<B>> Bot<B> for MuZeroBot<B, M> {
    fn select_move(&mut self, board: &B) -> B::Move {
        let tree = self
            .settings
            .build_tree(board, u32::MAX, &mut self.root_exec, &mut self.expand_exec, |tree| {
                tree.root_visits() >= self.visits
            });
        let index = tree.best_move_index().unwrap();
        // the root move index is always valid
        let mv = self.mapper.index_to_move(board, index).unwrap();
        mv
    }
}
