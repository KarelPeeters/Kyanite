use std::borrow::Borrow;
use std::ffi::OsStr;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use board_game::board::Board;

use cuda_nn_eval::executor::CudaGraphExecutor;
use cuda_nn_eval::graph::Graph;
use cuda_nn_eval::onnx::load_onnx_graph;
use cuda_sys::wrapper::handle::{CudaStream, CudnnHandle, Device};

use crate::mapping::BoardMapper;
use crate::network::{Network, ZeroEvaluation};
use crate::network::decode_policy::decode_output;

pub struct CudnnNetwork<B: Board, M: BoardMapper<B>> {
    mapper: M,
    path: PathBuf,
    max_batch_size: usize,

    executor: CudaGraphExecutor,

    input: Vec<f32>,
    wdl_logit: Vec<f32>,
    policy_logit: Vec<f32>,

    ph: PhantomData<B>,
}

impl<B: Board, M: BoardMapper<B>> CudnnNetwork<B, M> {
    pub fn load(mapper: M, path: impl AsRef<Path>, max_batch_size: usize, device: Device) -> Self {
        let path = path.as_ref().to_owned();
        assert_eq!(Some(OsStr::new("onnx")), path.extension(), "Unexpected extension");

        let graph = load_onnx_graph(&path, max_batch_size as i32);
        Self::check_shapes(max_batch_size as i32, &graph);

        let handle = CudnnHandle::new(CudaStream::new(device));
        let executor = CudaGraphExecutor::new(handle, &graph);

        let input = vec![0.0; max_batch_size * M::INPUT_SIZE];
        let wdl_logit = vec![0.0; max_batch_size * 3];
        let policy_logit = vec![0.0; max_batch_size * M::POLICY_SIZE];

        CudnnNetwork { path, max_batch_size, mapper, executor, input, wdl_logit, policy_logit, ph: PhantomData }
    }

    fn check_shapes(batch_size: i32, graph: &Graph) {
        // input
        let inputs = graph.inputs();
        assert_eq!(1, inputs.len(), "Wrong number of inputs");
        let [input_n, input_c, input_w, input_h] = graph[inputs[0]].shape;
        assert_eq!(batch_size, input_n);
        assert_eq!(M::INPUT_SIZE as i32, input_c * input_w * input_h, "Unexpected input size {:?}", graph[inputs[0]].shape);

        // outputs
        let outputs = graph.outputs();
        assert_eq!(2, outputs.len(), "Wrong number of outputs, expected wdl and policy");
        let wdl = outputs[0];
        let policy = outputs[1];

        // wdl
        let [wdl_n, wdl_c, wdl_w, wdl_h] = graph[wdl].shape;
        assert_eq!(batch_size, wdl_n);
        assert!(wdl_c == 3 && wdl_w == 1 && wdl_h == 1, "Unexpected wdl shape {:?}", graph[wdl].shape);

        // policy
        let [policy_n, policy_c, policy_w, policy_h] = graph[policy].shape;
        assert_eq!(batch_size, policy_n);
        assert_eq!(M::POLICY_SIZE as i32, policy_c * policy_w * policy_h, "Unexpected policy size {:?}", graph[policy].shape);
    }
}

impl<B: Board, M: BoardMapper<B>> Network<B> for CudnnNetwork<B, M> {
    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation> {
        let batch_size = boards.len();
        let max_batch_size = self.max_batch_size;
        assert!(batch_size <= max_batch_size);

        // encode input
        self.input.clear();
        for board in boards {
            self.mapper.append_board_to(&mut self.input, board.borrow())
        }

        // fill rest of input with zeros
        self.input.resize(max_batch_size * M::INPUT_SIZE, f32::NAN);

        // run the actual computation
        self.executor.run(&[&self.input], &mut [&mut self.wdl_logit, &mut self.policy_logit]);

        // decode the relevant part of the output
        decode_output(
            self.mapper,
            boards,
            &self.wdl_logit[0..batch_size * 3],
            &self.policy_logit[0..batch_size * M::POLICY_SIZE],
        )
    }
}

impl<B: Board, M: BoardMapper<B>> Debug for CudnnNetwork<B, M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudnnNetwork")
            .field("mapper", &self.mapper)
            .field("path", &self.path)
            .field("max_batch_size", &self.max_batch_size)
            .finish()
    }
}
