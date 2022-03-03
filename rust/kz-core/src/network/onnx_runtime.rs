use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use board_game::board::Board;

use onnxruntime::{GraphOptimizationLevel, LoggingLevel};
use onnxruntime::environment::Environment;
use onnxruntime::ndarray::Array;
use onnxruntime::ndarray::IxDyn;
use onnxruntime::session::Session;
use onnxruntime::tensor::OrtOwnedTensor;
use self_cell::self_cell;

use crate::mapping::BoardMapper;
use crate::network::{Network, ZeroEvaluation};
use crate::network::common::decode_output;

pub struct OnnxNetwork<B: Board, M: BoardMapper<B>> {
    mapper: M,
    path: PathBuf,
    inner: Inner,
    ph: PhantomData<B>,
}

impl<B: Board, M: BoardMapper<B>> Debug for OnnxNetwork<B, M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxNetwork")
            .field("mapper", &self.mapper)
            .field("path", &self.path)
            .field("inner", &self.inner)
            .finish()
    }
}

self_cell!(
    struct Inner {
        owner: Environment,
        #[covariant]
        dependent: Session,
    }

    impl {Debug}
);

impl<B: Board, M: BoardMapper<B>> OnnxNetwork<B, M> {
    pub fn load(mapper: M, path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_owned();
        let path_clone = path.clone();

        let env = Environment::builder()
            .with_log_level(LoggingLevel::Verbose)
            .build()
            .expect("Failed to build environment");

        let inner = Inner::new(env, move |env| {
            env.new_session_builder()
                .expect("Failed to create session builder")
                .with_optimization_level(GraphOptimizationLevel::All)
                .expect("Failed to set graph optimization level")
                .with_model_from_file(path_clone)
                .expect("Failed to build session")
        });

        OnnxNetwork { mapper, path, inner, ph: PhantomData }
    }
}

impl<B: Board, M: BoardMapper<B>> Network<B> for OnnxNetwork<B, M> {
    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation> {
        let batch_size = boards.len();

        let mut input = vec![];
        for board in boards {
            self.mapper.encode_full(&mut input, board.borrow())
        }

        let input_shape = (batch_size, M::INPUT_FULL_PLANES, M::INPUT_BOARD_SIZE, M::INPUT_BOARD_SIZE);
        let policy_shape = [batch_size, M::POLICY_PLANES, M::POLICY_PLANES, M::POLICY_PLANES];

        let input = Array::from_shape_vec(input_shape, input)
            .expect("Dimension mismatch");
        let input = vec![input];

        let mapper = self.mapper;
        self.inner.with_dependent_mut(|_, session: &mut Session| {
            let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(input)
                .expect("Session run failed");

            assert_eq!(3, outputs.len());
            let value = &outputs[0];
            let wdl = &outputs[0];
            let policy = &outputs[0];

            assert_eq!(IxDyn(&[batch_size]), value.dim());
            assert_eq!(IxDyn(&[batch_size, 3]), wdl.dim());
            assert_eq!(IxDyn(&policy_shape), policy.dim());

            let value = value.as_slice().unwrap();
            let wdl = wdl.as_slice().unwrap();
            let policy = policy.as_slice().unwrap();

            decode_output(mapper, &boards, value, wdl, policy)
        })
    }
}

