use std::borrow::Cow;
use std::fs::File;
use std::marker::PhantomData;

use board_game::board::Board;
use itertools::Itertools;
use serde::Deserialize;

use cuda_nn_eval::executor::CudaExecutor;
use cuda_nn_eval::quant::QuantizedStorage;
use cuda_sys::wrapper::handle::Device;
use kz_util::Pad;
use nn_graph::graph::{Graph, SliceRange};
use nn_graph::onnx::load_graph_from_onnx_path;
use nn_graph::optimizer::{optimize_graph, OptimizerSettings};
use nn_graph::shape;
use nn_graph::shape::{Shape, Size};

use crate::mapping::BoardMapper;
use crate::muzero::MuZeroEvaluation;
use crate::network::common::{softmax_in_place, zero_values_from_scalars};

#[derive(Deserialize, Debug, Clone)]
pub struct MuZeroNetworkInfo {
    pub game: String,
    pub state_channels: usize,
    pub state_channels_saved: usize,
    pub state_quant_bits: u8,
}

impl MuZeroNetworkInfo {
    pub fn state_shape<B: Board, M: BoardMapper<B>>(&self, mapper: M) -> Shape {
        let b = mapper.state_board_size();
        shape![Size::BATCH, self.state_channels, b, b]
    }

    pub fn state_saved_shape<B: Board, M: BoardMapper<B>>(&self, mapper: M) -> Shape {
        let b = mapper.state_board_size();
        shape![Size::BATCH, self.state_channels_saved, b, b]
    }
}

pub struct MuZeroGraphs<B: Board, M: BoardMapper<B>> {
    pub mapper: M,
    pub info: MuZeroNetworkInfo,

    pub representation: Graph,
    pub dynamics: Graph,
    pub prediction: Graph,

    pub ph: PhantomData<B>,
}

pub struct MuZeroFusedGraphs<B: Board, M: BoardMapper<B>> {
    pub mapper: M,
    pub info: MuZeroNetworkInfo,

    pub root: Graph,
    pub expand: Graph,

    pub ph: PhantomData<B>,
}

pub struct MuZeroFusedExecutors<B: Board, M: BoardMapper<B>> {
    pub mapper: M,
    pub info: MuZeroNetworkInfo,

    pub root_exec: CudaExecutor,
    pub expand_exec: CudaExecutor,

    input_buffer: Vec<f32>,
    output_scalars_buffer: Vec<f32>,
    output_policy_buffer: Vec<f32>,

    pub ph: PhantomData<B>,
}

impl<B: Board, M: BoardMapper<B>> MuZeroGraphs<B, M> {
    pub fn load(path: &str, mapper: M) -> MuZeroGraphs<B, M> {
        assert!(path.ends_with("_"), "Path should end with '_', got '{}'", path);

        let info: MuZeroNetworkInfo =
            serde_json::from_reader(File::open(format!("{}info.json", path)).unwrap()).expect("Failed to parse info");
        assert_eq!(info.state_quant_bits, 8, "Only 8-bit quantization supported for now");

        let representation = load_graph_from_onnx_path(format!("{}representation.onnx", path));
        let dynamics = load_graph_from_onnx_path(format!("{}dynamics.onnx", path));
        let prediction = load_graph_from_onnx_path(format!("{}prediction.onnx", path));

        let input_shape = shape![Size::BATCH].concat(&Shape::fixed(&mapper.input_full_shape()));
        let state_shape = info.state_shape(mapper);
        let state_saved_shape = info.state_saved_shape(mapper);
        let action_shape = shape![Size::BATCH].concat(&Shape::fixed(&mapper.encoded_move_shape()));
        let policy_shape = shape![Size::BATCH].concat(&Shape::fixed(&mapper.policy_shape()));
        let scalar_shape = shape![Size::BATCH, 5];

        assert_eq!(representation.input_shapes(), &[input_shape]);
        assert_eq!(representation.output_shapes(), &[state_shape.clone()]);
        assert_eq!(dynamics.input_shapes(), &[state_saved_shape, action_shape]);
        assert_eq!(dynamics.output_shapes(), &[state_shape.clone()]);
        assert_eq!(prediction.input_shapes(), &[state_shape.clone()]);
        assert_eq!(prediction.output_shapes(), &[scalar_shape, policy_shape]);

        MuZeroGraphs {
            mapper,
            info,
            representation,
            dynamics,
            prediction,
            ph: PhantomData,
        }
    }

    pub fn optimize(&self, settings: OptimizerSettings) -> MuZeroGraphs<B, M> {
        MuZeroGraphs {
            mapper: self.mapper,
            info: self.info.clone(),
            representation: optimize_graph(&self.representation, settings),
            dynamics: optimize_graph(&self.dynamics, settings),
            prediction: optimize_graph(&self.prediction, settings),
            ph: Default::default(),
        }
    }

    pub fn fuse(&self, settings: OptimizerSettings) -> MuZeroFusedGraphs<B, M> {
        let root = {
            let mut root = Graph::new();

            let input_shape = Shape::fixed(&self.mapper.input_full_shape()).batched();

            let input = root.input(input_shape);
            let state = root.call(&self.representation, &[input])[0];
            let state_saved = root.slice(state, 1, SliceRange::simple(0, self.info.state_channels_saved));
            let outputs = root.call(&self.prediction, &[state]);
            root.output_all(&[state_saved, outputs[0], outputs[1]]);

            root
        };

        let expand = {
            let mut expand = Graph::new();
            let b = self.mapper.state_board_size();

            let prev_state = expand.input(shape![Size::BATCH, self.info.state_channels_saved, b, b]);
            let mv = expand.input(Shape::fixed(&self.mapper.encoded_move_shape()).batched());
            let state = expand.call(&self.dynamics, &[prev_state, mv])[0];
            let state_saved = expand.slice(state, 1, SliceRange::simple(0, self.info.state_channels_saved));
            let outputs = expand.call(&self.prediction, &[state]);
            expand.output_all(&[state_saved, outputs[0], outputs[1]]);

            expand
        };

        MuZeroFusedGraphs {
            mapper: self.mapper,
            info: self.info.clone(),
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
            info: self.info.clone(),

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
    pub fn eval_root(&mut self, boards: &[B]) -> Vec<(QuantizedStorage, MuZeroEvaluation<'static>)> {
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

    pub fn eval_expand(
        &mut self,
        pairs: &[(QuantizedStorage, usize)],
    ) -> Vec<(QuantizedStorage, MuZeroEvaluation<'static>)> {
        let batch_size = self.expand_exec.batch_size;

        assert!(
            pairs.len() <= batch_size,
            "Batch size is {}, but got {} boards",
            batch_size,
            pairs.len()
        );

        unsafe {
            let stream = self.expand_exec.handles.cudnn.stream();

            // copy and encode inputs
            self.input_buffer.clear();
            for (i, &(ref prev_state, mv_index)) in pairs.iter().enumerate() {
                prev_state.launch_copy_to_simple_tensor(&self.expand_exec.inputs[0].index(0, i), stream);
                self.mapper.encode_mv(&mut self.input_buffer, mv_index);
            }
            self.input_buffer
                .pad(self.mapper.encoded_mv_len() * batch_size, f32::NAN);
            self.expand_exec.inputs[1].copy_from_host_staged(&self.input_buffer);

            // run model (on the same stream as the quantizations, so no sync necessary)
            self.expand_exec.run_async();
            self.expand_exec.handles.cudnn.stream().synchronize();

            // copy outputs back
            self.root_exec.outputs[1].copy_simple_to_host(&mut self.output_scalars_buffer);
            self.root_exec.outputs[2].copy_simple_to_host(&mut self.output_policy_buffer);

            // decode outputs
            self.decode_outputs(pairs.len(), false)
        }
    }

    unsafe fn decode_outputs(
        &mut self,
        count: usize,
        is_root: bool,
    ) -> Vec<(QuantizedStorage, MuZeroEvaluation<'static>)> {
        // get the right executor
        let exec = if is_root { &self.root_exec } else { &self.expand_exec };
        let device = exec.handles.cudnn.device();
        let stream = exec.handles.cudnn.stream();

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
        let result = (0..count)
            .map(|bi| {
                let state = states.index(0, bi);
                let state_quant = QuantizedStorage::alloc(device, state.shape.size());
                state_quant.launch_copy_from_simple_tensor(&state, stream);

                let scalars = &self.output_scalars_buffer[5 * bi..5 * (bi + 1)];
                let mut policy = self.output_policy_buffer[policy_len * bi..policy_len * (bi + 1)].to_vec();

                softmax_in_place(&mut policy);

                let eval = MuZeroEvaluation {
                    values: zero_values_from_scalars(scalars),
                    policy: Cow::Owned(policy),
                };

                (state_quant, eval)
            })
            .collect_vec();

        // wait for quantizations to complete
        stream.synchronize();

        result
    }
}
