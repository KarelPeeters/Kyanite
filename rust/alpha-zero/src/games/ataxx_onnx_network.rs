use std::path::Path;

use onnxruntime::environment::Environment;
use onnxruntime::session::Session;
use self_cell::self_cell;
use onnxruntime::{LoggingLevel, GraphOptimizationLevel};
use crate::network::Network;
use board_game::games::ataxx::AtaxxBoard;
use crate::games::ataxx_utils::{encode_input, decode_output};
use onnxruntime::ndarray::Array;
use onnxruntime::tensor::OrtOwnedTensor;
use onnxruntime::ndarray::IxDyn;
use crate::zero::ZeroEvaluation;
use cuda_nn_eval::util::WrapDebug;

#[derive(Debug)]
pub struct AtaxxOnnxNetwork {
    inner: WrapDebug<Inner>,
}

self_cell!(
    struct Inner {
        owner: Environment,

        #[covariant]
        dependent: Session,
    }
);

impl AtaxxOnnxNetwork {
    pub fn load(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_owned();

        let env = Environment::builder()
            .with_log_level(LoggingLevel::Verbose)
            .build()
            .expect("Failed to build environment");

        let inner = Inner::new(env, move |env| {
            env.new_session_builder()
                .expect("Failed to create session builder")
                .with_optimization_level(GraphOptimizationLevel::All)
                .expect("Failed to set graph optimization level")
                .with_model_from_file(path)
                .expect("Failed to build session")
        });

        AtaxxOnnxNetwork { inner: inner.into() }
    }
}

impl Network<AtaxxBoard> for AtaxxOnnxNetwork {
    fn evaluate_batch(&mut self, boards: &[AtaxxBoard]) -> Vec<ZeroEvaluation> {
        let batch_size = boards.len();

        let input = encode_input(boards);
        let input = Array::from_shape_vec((batch_size, 5, 9, 9), input)
            .expect("Dimension mismatch");
        let input = vec![input];

        self.inner.with_dependent_mut(|_, session| {
            let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input)
                .expect("Failed to call session.run");

            assert_eq!(2, outputs.len(), "unexpected output count");
            let value = &outputs[0];
            let policy = &outputs[1];

            assert_eq!(value.dim(), IxDyn(&[batch_size, 3]));
            assert_eq!(policy.dim(), IxDyn(&[batch_size, 17, 9, 9]));

            let wdl = value.as_slice().unwrap();
            let policy = policy.as_slice().unwrap();

            decode_output(boards, wdl, policy)
        })
    }
}
