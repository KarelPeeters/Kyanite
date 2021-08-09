use std::ffi::OsStr;
use std::path::Path;



use board_game::games::ataxx::AtaxxBoard;
use cuda_nn_eval::executor::CudaGraphExecutor;

use cuda_nn_eval::onnx::load_onnx_graph;
use cuda_sys::wrapper::handle::{CudaStream, CudnnHandle, Device};

use crate::games::ataxx_utils::{decode_output, encode_input, INPUT_SIZE, POLICY_SIZE};
use crate::network::Network;
use crate::zero::ZeroEvaluation;

#[derive(Debug)]
pub struct AtaxxCNNNetwork {
    executor: CudaGraphExecutor,
    max_batch_size: usize,
}

impl AtaxxCNNNetwork {
    pub fn load(path: impl AsRef<Path>, max_batch_size: usize, device: Device) -> Self {
        let path = path.as_ref();
        assert_eq!(Some(OsStr::new("onnx")), path.extension(), "Unexpected extension");

        let graph = load_onnx_graph(path, max_batch_size as i32);
        let handle = CudnnHandle::new(CudaStream::new(device));
        let executor = CudaGraphExecutor::new(handle, &graph);

        AtaxxCNNNetwork { executor, max_batch_size }
    }
}

impl Network<AtaxxBoard> for AtaxxCNNNetwork {
    fn evaluate_batch(&mut self, boards: &[AtaxxBoard]) -> Vec<ZeroEvaluation> {
        let batch_size = boards.len();
        let max_batch_size = self.max_batch_size;
        assert!(batch_size <= max_batch_size);

        //append zeros until we reach the max batch size for which the executor was designed
        let mut input = encode_input(boards);
        input.resize(max_batch_size * INPUT_SIZE, f32::NAN);

        let mut output_wdl_logit = vec![0.0; max_batch_size * 3];
        let mut output_policy_logit = vec![0.0; max_batch_size * POLICY_SIZE];

        self.executor.run(&[&input], &mut [&mut output_wdl_logit, &mut output_policy_logit]);

        // only keep the useful output
        let output_wdl_logit = &output_wdl_logit[0..batch_size * 3];
        let output_policy_logit = &output_policy_logit[0..batch_size * POLICY_SIZE];

        decode_output(boards, output_wdl_logit, output_policy_logit)
    }
}