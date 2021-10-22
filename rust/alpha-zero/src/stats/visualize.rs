use std::borrow::Borrow;

use board_game::board::Board;
use onnxruntime::tensor::ndarray_tensor::NdArrayTensor;

use nn_graph::cpu::Tensor;
use nn_graph::graph::Value;
use nn_graph::ndarray::{Array2, Axis};
use nn_graph::visualize::{Image, visualize_graph_activations};

use crate::mapping::BoardMapper;
use crate::network::cpu::CPUNetwork;
use crate::util::IndexOf;

pub fn visualize_network_activations<B: Board, M: BoardMapper<B>>(
    network: &mut CPUNetwork<B, M>,
    boards: &[impl Borrow<B>],
) -> Vec<Image> {
    let execution = network.evaluate_batch_exec(boards);

    let graph = network.graph();
    let mapper = network.mapper();
    assert_eq!(graph.outputs().len(), 3);

    let post_process = |value: Value, tensor: Tensor| {
        match graph.outputs().iter().index_of(&value) {
            None => None,
            Some(0) => Some(tensor.mapv(f32::tanh).to_shared()),
            Some(1) => Some(tensor.softmax(Axis(1)).to_shared()),
            Some(2) => {
                let mut result_logit: Array2<f32> = tensor.reshape((boards.len(), M::POLICY_SIZE)).to_owned();
                for (bi, board) in boards.iter().enumerate() {
                    let board = board.borrow();

                    for i in 0..M::POLICY_SIZE {
                        let is_available = mapper.index_to_move(board, i)
                            .map_or(false, |mv| board.is_available_move(mv));

                        if !is_available {
                            result_logit[(bi, i)] = f32::NEG_INFINITY;
                        }
                    }
                }
                let result = result_logit.softmax(Axis(1))
                    .into_shape([&[boards.len()][..], &M::POLICY_SHAPE].concat())
                    .unwrap().to_shared();
                Some(result)
            }
            _ => unreachable!(),
        }
    };

    visualize_graph_activations(&graph, &execution, post_process)
}