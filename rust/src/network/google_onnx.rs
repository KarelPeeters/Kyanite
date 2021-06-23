use std::path::Path;

use onnxruntime::{GraphOptimizationLevel, LoggingLevel};
use onnxruntime::environment::Environment;
use onnxruntime::ndarray::Array;
use onnxruntime::ndarray::IxDyn;
use onnxruntime::session::Session;
use onnxruntime::tensor::OrtOwnedTensor;
use self_cell::self_cell;
use sttt::board::Board;

use crate::network::{collect_google_output, encode_google_input, Network, NetworkEvaluation};

pub struct GoogleOnnxNetwork {
    inner: Inner,
}

self_cell!(
    struct Inner {
        owner: Environment,

        #[covariant]
        dependent: Session,
    }
);

impl GoogleOnnxNetwork {
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

        GoogleOnnxNetwork { inner }
    }
}

impl Network for GoogleOnnxNetwork {
    fn evaluate_batch(&mut self, boards: &[Board]) -> Vec<NetworkEvaluation> {
        let batch_size = boards.len();

        let input = encode_google_input(boards);
        let input = Array::from_shape_vec((batch_size, 5, 9, 9), input)
            .expect("Dimension mismatch");
        let input = vec![input];

        self.inner.with_dependent_mut(|_, session| {
            let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input)
                .expect("Failed to call session.run");

            assert_eq!(2, outputs.len(), "unexpected output count");
            let value = &outputs[0];
            let policy = &outputs[1];

            assert_eq!(value.dim(), IxDyn(&[batch_size, 1]));
            assert_eq!(policy.dim(), IxDyn(&[batch_size, 81]));

            let value = value.as_slice().unwrap();
            let policy = policy.as_slice().unwrap();
            collect_google_output(boards, value, policy)
        })
    }
}
