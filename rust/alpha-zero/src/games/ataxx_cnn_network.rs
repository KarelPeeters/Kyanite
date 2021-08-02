use std::ffi::OsStr;
use std::path::Path;

use npyz::npz::NpzArchive;

use board_game::games::ataxx::AtaxxBoard;
use cuda_nn_eval::executor::CudaGraphExecutor;
use cuda_nn_eval::graph::Graph;
use cuda_nn_eval::load::load_params_from_npz;
use cuda_sys::wrapper::handle::{CudaStream, CudnnHandle, Device};

use crate::games::ataxx_utils::{decode_output, encode_input, INPUT_SIZE, POLICY_SIZE};
use crate::network::Network;
use crate::selfplay::generate_zero::NetworkLoader;
use crate::zero::ZeroEvaluation;
use crate::network::tower_shape::TowerShape;

#[derive(Debug)]
pub struct AtaxxCNNLoader {
    pub path: String,
    pub shape: TowerShape,
    pub max_batch_size: usize,
}

impl NetworkLoader<AtaxxBoard> for AtaxxCNNLoader {
    type Device = Device;
    type Network = AtaxxCNNNetwork;

    fn load_network(&self, device: Self::Device) -> Self::Network {
        let graph = self.shape.to_graph(self.max_batch_size as i32);
        AtaxxCNNNetwork::load(&self.path, &graph, self.max_batch_size, device)
    }
}

#[derive(Debug)]
pub struct AtaxxCNNNetwork {
    executor: CudaGraphExecutor,
    max_batch_size: usize,
}

impl AtaxxCNNNetwork {
    pub fn load(path: impl AsRef<Path>, graph: &Graph, max_batch_size: usize, device: Device) -> Self {
        let path = path.as_ref();
        assert_eq!(Some(OsStr::new("npz")), path.extension(), "Unexpected extension");

        let mut npz = NpzArchive::open(path)
            .expect("Failed to open npz file");
        let params = load_params_from_npz(&graph, &mut npz, device);

        let handle = CudnnHandle::new(CudaStream::new(device));
        let executor = CudaGraphExecutor::new(handle, &graph, params);

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

        self.executor.eval(&[&input], &mut [&mut output_wdl_logit, &mut output_policy_logit]);

        // only keep the useful output
        let output_wdl_logit = &output_wdl_logit[0..batch_size * 3];
        let output_policy_logit = &output_policy_logit[0..batch_size * POLICY_SIZE];

        decode_output(boards, output_wdl_logit, output_policy_logit)
    }
}