use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;

use board_game::board::Board;
use itertools::Itertools;

use nn_graph::cpu::{cpu_execute_graph, ExecutionInfo, Tensor};
use nn_graph::graph::Graph;
use nn_graph::ndarray::IxDyn;

use crate::mapping::BoardMapper;
use crate::network::common::{check_graph_shapes, decode_output};
use crate::network::{Network, ZeroEvaluation};

pub struct CPUNetwork<B: Board, M: BoardMapper<B>> {
    mapper: M,
    graph: Graph,
    ph: PhantomData<B>,
}

impl<B: Board, M: BoardMapper<B>> CPUNetwork<B, M> {
    pub fn new(mapper: M, graph: Graph) -> Self {
        check_graph_shapes(mapper, &graph);

        CPUNetwork {
            mapper,
            graph,
            ph: Default::default(),
        }
    }

    pub fn evaluate_batch_exec(&mut self, boards: &[impl Borrow<B>]) -> ExecutionInfo {
        let batch_size = boards.len();

        // encore the input
        let mut input = vec![];
        for board in boards {
            self.mapper.encode_full(&mut input, board.borrow())
        }
        let input_len = input.len();

        let mut input_shape = vec![batch_size];
        input_shape.extend_from_slice(&self.mapper.input_full_shape());

        let input = Tensor::from_shape_vec(IxDyn(&input_shape), input)
            .unwrap_or_else(|_| panic!("Incompatible shapes: ({}) -> {:?}", input_len, input_shape));

        // evaluate the graph
        cpu_execute_graph(&self.graph, batch_size, &[input])
    }

    pub fn mapper(&self) -> M {
        self.mapper
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }
}

impl<B: Board, M: BoardMapper<B>> Network<B> for CPUNetwork<B, M> {
    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation<'static>> {
        let outputs = self.evaluate_batch_exec(boards).output_tensors();
        let batch_outputs = outputs.iter().map(|t| t.as_slice().unwrap()).collect_vec();

        decode_output(self.mapper, boards, &batch_outputs)
    }
}

impl<B: Board, M: BoardMapper<B>> Debug for CPUNetwork<B, M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CPUNetwork")
            .field("graph", &self.graph)
            .field("mapper", &self.mapper)
            .finish()
    }
}
