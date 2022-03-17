use std::borrow::Cow;
use std::marker::PhantomData;

use board_game::board::Board;
use itertools::Itertools;

use cuda_nn_eval::executor::CudaExecutor;
use cuda_nn_eval::tensor::DeviceTensor;
use cuda_sys::wrapper::handle::Device;
use kz_util::Pad;
use nn_graph::graph::Graph;
use nn_graph::optimizer::{optimize_graph, OptimizerSettings};
use nn_graph::shape;
use nn_graph::shape::{Shape, Size};

use crate::mapping::BoardMapper;
use crate::muzero::MuZeroEvaluation;
use crate::network::common::{softmax_in_place, zero_value_from_scalars};

pub struct MuZeroGraphs<B: Board, M: BoardMapper<B>> {
    pub mapper: M,
    pub representation: Graph,
    pub dynamics: Graph,
    pub prediction: Graph,

    pub ph: PhantomData<B>,
}

pub struct MuZeroFusedGraphs<B: Board, M: BoardMapper<B>> {
    pub mapper: M,
    pub state_shape: Shape,

    pub root: Graph,
    pub expand: Graph,

    pub ph: PhantomData<B>,
}

pub struct MuZeroFusedExecutors<B: Board, M: BoardMapper<B>> {
    pub mapper: M,
    pub state_shape: Shape,

    pub root_exec: CudaExecutor,
    pub expand_exec: CudaExecutor,

    input_buffer: Vec<f32>,
    output_scalars_buffer: Vec<f32>,
    output_policy_buffer: Vec<f32>,

    pub ph: PhantomData<B>,
}

impl<B: Board, M: BoardMapper<B>> MuZeroGraphs<B, M> {
    pub fn optimize(&self, settings: OptimizerSettings) -> MuZeroGraphs<B, M> {
        MuZeroGraphs {
            mapper: self.mapper,
            representation: optimize_graph(&self.representation, settings),
            dynamics: optimize_graph(&self.dynamics, settings),
            prediction: optimize_graph(&self.prediction, settings),
            ph: Default::default(),
        }
    }

    pub fn fuse(&self, settings: OptimizerSettings) -> MuZeroFusedGraphs<B, M> {
        let state_shape;
        let root = {
            let mut root = Graph::new();

            let [c, w, h] = self.mapper.input_full_shape();
            let input = root.input(shape![Size::BATCH, c, w, h]);

            let state = root.call(&self.representation, &[input]);
            assert_eq!(state.len(), 1);
            let state = state[0];
            state_shape = root[state].shape.clone();

            let outputs = root.call(&self.prediction, &[state]);
            assert_eq!(outputs.len(), 2);
            root.output_all(&[state, outputs[0], outputs[1]]);

            root
        };

        let expand = {
            let mut expand = Graph::new();

            let [c, w, h] = self.mapper.encoded_move_shape();

            let prev_state = expand.input(state_shape.clone());
            let mv = expand.input(shape![Size::BATCH, c, w, h]);

            let state = expand.call(&self.dynamics, &[prev_state, mv]);
            assert_eq!(state.len(), 1);
            let state = state[0];

            let outputs = expand.call(&self.prediction, &[state]);
            assert_eq!(outputs.len(), 2);
            expand.output_all(&[state, outputs[0], outputs[1]]);

            expand
        };

        MuZeroFusedGraphs {
            mapper: self.mapper,
            state_shape,
            root: optimize_graph(&root, settings),
            expand: optimize_graph(&expand, settings),
            ph: PhantomData,
        }
    }
}

impl<B: Board, M: BoardMapper<B>> MuZeroFusedGraphs<B, M> {
    pub fn executors(
        &self,
        device: Device,
        root_batch_size: usize,
        expand_batch_size: usize,
    ) -> MuZeroFusedExecutors<B, M> {
        MuZeroFusedExecutors {
            mapper: self.mapper,
            state_shape: self.state_shape.clone(),

            root_exec: CudaExecutor::new(device, &self.root, root_batch_size),
            expand_exec: CudaExecutor::new(device, &self.expand, expand_batch_size),

            input_buffer: vec![],
            output_scalars_buffer: vec![],
            output_policy_buffer: vec![],

            ph: Default::default(),
        }
    }
}

impl<B: Board, M: BoardMapper<B>> MuZeroFusedExecutors<B, M> {
    pub fn eval_root(&mut self, boards: &[B]) -> Vec<(DeviceTensor, MuZeroEvaluation<'static>)> {
        let batch_size = self.root_exec.batch_size;
        assert!(
            boards.len() <= batch_size,
            "Batch size is {}, but got {} boards",
            batch_size,
            boards.len()
        );

        // encode inputs, all padded until the batch size
        self.input_buffer.clear();
        for board in boards {
            self.mapper.encode_input_full(&mut self.input_buffer, board);
        }
        self.input_buffer
            .pad(self.mapper.input_full_len() * batch_size, f32::NAN);

        unsafe {
            // copy inputs
            self.root_exec.inputs[0].copy_simple_from_host(&self.input_buffer);

            // run model
            self.root_exec.run_async();
            self.expand_exec.handles.cudnn.stream().synchronize();

            // copy & decode outputs
            self.decode_outputs(boards.len(), true)
        }
    }

    pub fn eval_expand(&mut self, pairs: &[(DeviceTensor, usize)]) -> Vec<(DeviceTensor, MuZeroEvaluation<'static>)> {
        let batch_size = self.expand_exec.batch_size;
        assert!(
            pairs.len() <= batch_size,
            "Batch size is {}, but got {} boards",
            batch_size,
            pairs.len()
        );

        unsafe {
            // copy and encode inputs
            self.input_buffer.clear();
            for (i, &(ref prev_state, mv_index)) in pairs.iter().enumerate() {
                self.expand_exec.inputs[0].index(0, i).copy_from(prev_state);
                self.mapper.encode_mv(&mut self.input_buffer, mv_index);
            }
            self.input_buffer
                .pad(self.mapper.encoded_mv_len() * batch_size, f32::NAN);
            self.expand_exec.inputs[1].copy_from_host_staged(&self.input_buffer);

            // run model
            self.expand_exec.run_async();
            self.expand_exec.handles.cudnn.stream().synchronize();

            // copy outputs back
            self.root_exec.outputs[1].copy_simple_to_host(&mut self.output_scalars_buffer);
            self.root_exec.outputs[2].copy_simple_to_host(&mut self.output_policy_buffer);

            // decode outputs
            self.decode_outputs(pairs.len(), false)
        }
    }

    unsafe fn decode_outputs(&mut self, count: usize, is_root: bool) -> Vec<(DeviceTensor, MuZeroEvaluation<'static>)> {
        // get the right executor
        let exec = if is_root { &self.root_exec } else { &self.expand_exec };

        let policy_len = self.mapper.policy_len();
        let batch_size = exec.batch_size;

        // prepare output buffers
        self.output_scalars_buffer.clear();
        self.output_scalars_buffer.pad(5 * batch_size, f32::NAN);
        self.output_policy_buffer.clear();
        self.output_policy_buffer.pad(policy_len * batch_size, f32::NAN);

        // copy outputs back
        let states = &exec.outputs[0];
        exec.outputs[1].copy_simple_to_host(&mut self.output_scalars_buffer);
        exec.outputs[2].copy_simple_to_host(&mut self.output_policy_buffer);

        // decode outputs
        (0..count)
            .map(|bi| {
                let state = states.index(0, bi).deep_clone();

                let scalars = &self.output_scalars_buffer[5 * bi..5 * (bi + 1)];
                let mut policy = self.output_policy_buffer[policy_len * bi..policy_len * (bi + 1)].to_vec();

                softmax_in_place(&mut policy);

                let eval = MuZeroEvaluation {
                    values: zero_value_from_scalars(scalars),
                    policy: Cow::Owned(policy),
                };

                (state, eval)
            })
            .collect_vec()
    }
}
