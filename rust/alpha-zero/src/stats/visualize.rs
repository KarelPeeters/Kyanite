use std::borrow::Borrow;

use board_game::board::{Board, Player};
use itertools::Itertools;
use onnxruntime::tensor::ndarray_tensor::NdArrayTensor;

use nn_graph::cpu::Tensor;
use nn_graph::graph::Value;
use nn_graph::ndarray::{Array2, Axis};
use nn_graph::visualize::{Image, visualize_graph_activations};

use crate::mapping::BoardMapper;
use crate::network::cpu::CPUNetwork;
use crate::util::IndexOf;

pub fn visualize_network_activations<'a, B: Board, M: BoardMapper<B>>(
    network: &mut CPUNetwork<B, M>,
    boards: &'a [impl Borrow<B>],
) -> (Vec<Image>, Vec<Image>) {
    let boards = boards.iter().map(|b| b.borrow());

    let boards_a = boards.clone()
        .filter(|&b| b.next_player() == Player::A)
        .collect_vec();
    let boards_b = boards.clone()
        .filter(|&b| b.next_player() == Player::B)
        .collect_vec();

    let exec_a = network.evaluate_batch_exec(&boards_a);
    let exec_b = network.evaluate_batch_exec(&boards_b);

    let graph = network.graph();
    let mapper = network.mapper();
    assert_eq!(graph.outputs().len(), 3);

    let post_process = |boards: Vec<&'a B>| move |value: Value, tensor: Tensor| {
        match graph.outputs().iter().index_of(&value) {
            None => None,
            Some(0) => Some(tensor.mapv(f32::tanh).to_shared()),
            Some(1) => Some(tensor.softmax(Axis(1)).to_shared()),
            Some(2) => {
                let mut result_logit: Array2<f32> = tensor.reshape((boards.len(), M::POLICY_SIZE)).to_owned();
                for (bi, &board) in boards.iter().enumerate() {
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

    (
        visualize_graph_activations(&graph, &exec_a, post_process(boards_a)),
        visualize_graph_activations(&graph, &exec_b, post_process(boards_b)),
    )
}