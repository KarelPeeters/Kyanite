use std::borrow::Borrow;
use std::ffi::OsStr;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use board_game::board::Board;

use cuda_nn_eval::cpu_executor::CpuExecutor;
use cuda_nn_eval::onnx::load_onnx_graph;

use crate::mapping::BoardMapper;
use crate::network::{Network, ZeroEvaluation};
use crate::network::decode_policy::decode_output;

pub struct CPUNetwork<B: Board, M: BoardMapper<B>> {
    mapper: M,
    path: PathBuf,
    executor: CpuExecutor,
    max_batch_size: usize,
    ph: PhantomData<B>,
}

impl<B: Board, M: BoardMapper<B>> CPUNetwork<B, M> {
    pub fn load(mapper: M, path: impl AsRef<Path>, max_batch_size: usize) -> Self {
        let path = path.as_ref().to_owned();
        assert_eq!(Some(OsStr::new("onnx")), path.extension(), "Unexpected extension");

        //TODO this is kind of stupid, it should be easy to support any batch size on CPU
        let graph = load_onnx_graph(&path, max_batch_size as i32);
        let executor = CpuExecutor::new(&graph);

        CPUNetwork { mapper, path, executor, max_batch_size, ph: PhantomData }
    }
}

impl<B: Board, M: BoardMapper<B>> Network<B> for CPUNetwork<B, M> {
    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation> {
        let batch_size = boards.len();
        let max_batch_size = self.max_batch_size;
        assert!(batch_size <= max_batch_size);

        //encore the input
        let mut input = vec![];
        for board in boards {
            self.mapper.append_board_to(&mut input, board.borrow())
        }

        // fill rest of input with zeros
        assert_eq!(batch_size * M::INPUT_SIZE, input.len());
        input.resize(max_batch_size * M::INPUT_SIZE, f32::NAN);

        // do the actual computation
        let mut output_wdl_logit = vec![0.0; max_batch_size * 3];
        let mut output_policy_logit = vec![0.0; max_batch_size * M::POLICY_SIZE];
        self.executor.evaluate(&[&input], &mut [&mut output_wdl_logit, &mut output_policy_logit]);

        // decode the relevant part of the output
        decode_output(
            self.mapper,
            boards,
            &output_wdl_logit[0..batch_size * 3],
            &output_policy_logit[0..batch_size * M::POLICY_SIZE],
        )
    }
}

impl<B: Board, M: BoardMapper<B>> Debug for CPUNetwork<B, M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuNetwork")
            .field("path", &self.path)
            .field("max_batch_size", &self.max_batch_size)
            .field("mapper", &self.mapper)
            .finish()
    }
}