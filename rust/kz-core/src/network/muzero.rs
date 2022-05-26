use std::borrow::Cow;
use std::fs::File;
use std::marker::PhantomData;

use board_game::board::Board;
use itertools::Itertools;
use serde::Deserialize;

use cuda_nn_eval::executor::CudaExecutor;
use cuda_nn_eval::quant::{BatchQuantizer, QuantizedStorage};
use cuda_sys::wrapper::handle::Device;
use kz_util::sequence::Pad;
use nn_graph::graph::{Graph, SliceRange};
use nn_graph::onnx::load_graph_from_onnx_path;
use nn_graph::optimizer::{optimize_graph, OptimizerSettings};
use nn_graph::shape;
use nn_graph::shape::{Shape, Size};

use crate::mapping::BoardMapper;
use crate::muzero::MuZeroEvaluation;
use crate::network::common::zero_values_from_scalars;
use crate::network::job_channel::JobClient;

pub type RootClient<B> = JobClient<RootArgs<B>, MuZeroEvaluation<'static>>;
pub type ExpandClient = JobClient<ExpandArgs, MuZeroEvaluation<'static>>;

#[derive(Deserialize, Debug, Clone, Eq, PartialEq)]
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

#[derive(Debug)]
pub struct MuZeroRootExecutor<B: Board, M: BoardMapper<B>> {
    pub mapper: M,
    ph: PhantomData<B>,

    pub info: MuZeroNetworkInfo,
    pub root_exec: CudaExecutor,

    input_buffer: Vec<f32>,
    output_decoder: MuZeroOutputDecoder<B, M>,
}

// TODO test whether these executors work properly with the wrong batch size
#[derive(Debug)]
pub struct MuZeroExpandExecutor<B: Board, M: BoardMapper<B>> {
    pub mapper: M,
    ph: PhantomData<B>,

    pub info: MuZeroNetworkInfo,
    pub expand_exec: CudaExecutor,

    input_buffer: Vec<f32>,
    output_decoder: MuZeroOutputDecoder<B, M>,
}

#[derive(Debug)]
pub struct MuZeroOutputDecoder<B: Board, M: BoardMapper<B>> {
    mapper: M,
    _info: MuZeroNetworkInfo,
    ph: PhantomData<B>,

    quantizer: BatchQuantizer,

    output_scalars_buffer: Vec<f32>,
    output_policy_buffer: Vec<f32>,
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
        let state_slice_range = SliceRange::simple(0, self.info.state_channels_saved);

        let root = {
            let mut root = Graph::new();

            let input_shape = Shape::fixed(&self.mapper.input_full_shape()).batched();

            let input = root.input(input_shape);
            let state = root.call(&self.representation, &[input])[0];
            let state_saved = root.slice(state, 1, state_slice_range);
            let outputs = root.call(&self.prediction, &[state]);
            root.output_all(&[state_saved, outputs[0], outputs[1]]);

            root
        };

        let expand = {
            let mut expand = Graph::new();

            let prev_state = expand.input(self.info.state_saved_shape(self.mapper));
            let mv = expand.input(Shape::fixed(&self.mapper.encoded_move_shape()).batched());
            let state = expand.call(&self.dynamics, &[prev_state, mv])[0];
            let state_saved = expand.slice(state, 1, state_slice_range);
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
    pub fn root_executor(&self, device: Device, max_batch_size: usize) -> MuZeroRootExecutor<B, M> {
        MuZeroRootExecutor {
            mapper: self.mapper,
            ph: Default::default(),
            info: self.info.clone(),

            root_exec: CudaExecutor::new(device, &self.root, max_batch_size),

            input_buffer: vec![],
            output_decoder: MuZeroOutputDecoder::new(device, max_batch_size, self.mapper, self.info.clone()),
        }
    }

    pub fn expand_executor(&self, device: Device, max_batch_size: usize) -> MuZeroExpandExecutor<B, M> {
        MuZeroExpandExecutor {
            mapper: self.mapper,
            ph: Default::default(),
            info: self.info.clone(),

            expand_exec: CudaExecutor::new(device, &self.expand, max_batch_size),

            input_buffer: vec![],
            output_decoder: MuZeroOutputDecoder::new(device, max_batch_size, self.mapper, self.info.clone()),
        }
    }
}

#[derive(Debug)]
pub struct ExpandArgs {
    pub state: QuantizedStorage,
    pub move_index: usize,
    pub output_state: QuantizedStorage,
}

#[derive(Debug)]
pub struct RootArgs<B> {
    pub board: B,
    pub output_state: QuantizedStorage,
}

impl<B: Board, M: BoardMapper<B>> MuZeroRootExecutor<B, M> {
    pub fn eval_root(&mut self, args: &[RootArgs<B>]) -> Vec<MuZeroEvaluation<'static>> {
        let max_batch_size = self.root_exec.batch_size;
        let batch_size = args.len();

        assert!(
            batch_size <= max_batch_size,
            "Max batch size is {}, but got {} boards",
            max_batch_size,
            batch_size
        );

        // encode inputs, all padded until the batch size
        self.input_buffer.clear();
        for arg in args {
            self.mapper.encode_input_full(&mut self.input_buffer, &arg.board);
        }
        self.input_buffer
            .pad(self.mapper.input_full_len() * max_batch_size, f32::NAN);

        unsafe {
            // copy inputs
            self.root_exec.inputs[0].copy_simple_from_host(&self.input_buffer);

            // run model
            self.root_exec.run_async();

            // get the result
            self.output_decoder.copy_and_decode_outputs(
                &mut self.root_exec,
                batch_size,
                args.iter().map(|a| &a.output_state),
            )
        }
    }
}

impl<B: Board, M: BoardMapper<B>> MuZeroExpandExecutor<B, M> {
    pub fn eval_expand(&mut self, args: &[ExpandArgs]) -> Vec<MuZeroEvaluation<'static>> {
        let max_batch_size = self.expand_exec.batch_size;
        let batch_size = args.len();
        check_batch_size(batch_size, max_batch_size);

        unsafe {
            // encode inputs
            self.input_buffer.clear();
            for arg in args {
                self.mapper.encode_mv(&mut self.input_buffer, arg.move_index);
            }

            // copy inputs to device mem
            let max_input_size = self.mapper.encoded_mv_len() * max_batch_size;
            self.input_buffer.pad(max_input_size, f32::NAN);
            self.expand_exec.inputs[1].copy_from_host_staged(&self.input_buffer);

            // unquantize inputs (using the output decoder's quantizer, it's not doing anything else yet)
            let stream = self.expand_exec.handles.cudnn.stream();
            self.output_decoder.quantizer.launch_unquantize(
                stream,
                args.iter().map(|a| &a.state),
                &self.expand_exec.inputs[0],
            );

            // run model (on the same stream as the quantizations, so no sync necessary)
            self.expand_exec.run_async();

            // get the result
            let output_quants = args.iter().map(|a| &a.output_state);
            self.output_decoder
                .copy_and_decode_outputs(&mut self.expand_exec, batch_size, output_quants)
        }
    }
}

impl<B: Board, M: BoardMapper<B>> MuZeroOutputDecoder<B, M> {
    fn new(device: Device, max_batch_size: usize, mapper: M, info: MuZeroNetworkInfo) -> Self {
        Self {
            mapper,
            _info: info,
            ph: PhantomData,

            quantizer: BatchQuantizer::new(device, max_batch_size),
            output_scalars_buffer: vec![],
            output_policy_buffer: vec![],
        }
    }

    unsafe fn copy_and_decode_outputs<'a>(
        &mut self,
        exec: &mut CudaExecutor,
        batch_size: usize,
        output_quants: impl ExactSizeIterator<Item = &'a QuantizedStorage>,
    ) -> Vec<MuZeroEvaluation<'static>> {
        assert_eq!(output_quants.len(), batch_size);

        let stream = exec.handles.cudnn.stream();
        let policy_len = self.mapper.policy_len();

        // prepare output buffers
        self.output_scalars_buffer.clear();
        self.output_scalars_buffer.pad(5 * batch_size, f32::NAN);
        self.output_policy_buffer.clear();
        self.output_policy_buffer.pad(policy_len * batch_size, f32::NAN);

        // copy outputs back
        stream.synchronize();

        let slice = SliceRange::simple(0, batch_size);
        let device_states = exec.outputs[0].slice(0, slice);
        let device_scalars = exec.outputs[1].slice(0, slice);
        let device_policy = exec.outputs[2].slice(0, slice);

        device_scalars.copy_simple_to_host(&mut self.output_scalars_buffer);
        device_policy.copy_simple_to_host(&mut self.output_policy_buffer);

        self.quantizer.launch_quantize(&stream, &device_states, output_quants);

        // decode outputs
        let result = (0..batch_size)
            .map(|bi| {
                let scalars = &self.output_scalars_buffer[5 * bi..5 * (bi + 1)];
                let policy_logits = self.output_policy_buffer[policy_len * bi..policy_len * (bi + 1)].to_vec();

                MuZeroEvaluation {
                    values: zero_values_from_scalars(scalars),
                    policy_logits: Cow::Owned(policy_logits),
                }
            })
            .collect_vec();

        // wait for quantizations to complete
        stream.synchronize();

        result
    }
}

fn check_batch_size(actual: usize, max: usize) {
    assert!(actual <= max, "Max batch size is {}, but got {}", max, actual);
}
