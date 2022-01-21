use board_game::board::{Board, Player};

use nn_graph::cpu::{softmax, Tensor};
use nn_graph::graph::Value;
use nn_graph::ndarray::{Array2, Axis};
use nn_graph::visualize::{Image, visualize_graph_activations};

use crate::mapping::BoardMapper;
use crate::network::cpu::CPUNetwork;
use crate::util::IndexOf;

pub fn visualize_network_activations_split<B: Board, M: BoardMapper<B>>(
    network: &mut CPUNetwork<B, M>,
    boards: &[B],
    max_images: Option<usize>,
) -> (Vec<Image>, Vec<Image>) {
    let (boards_a, boards_b) = split_player(boards);

    (
        visualize_network_activations(network, &boards_a, max_images),
        visualize_network_activations(network, &boards_b, max_images),
    )
}

fn split_player<B: Board>(boards: &[B]) -> (Vec<B>, Vec<B>) {
    let mut result_a = vec![];
    let mut result_b = vec![];

    for board in boards {
        match board.next_player() {
            Player::A => { result_a.push(board.clone()) }
            Player::B => { result_b.push(board.clone()) }
        }
    }

    (result_a, result_b)
}

//TODO try to run all of this on the GPU instead, and see if that's faster
//  the basic idea is to add all tensors we want to plot as extra graph outputs
//  usually this will not be all of them, and we can decide in advance by looking at the shapes
pub fn visualize_network_activations<B: Board, M: BoardMapper<B>>(
    network: &mut CPUNetwork<B, M>,
    boards: &[B],
    max_images: Option<usize>,
) -> Vec<Image> {
    let exec = network.evaluate_batch_exec(boards);

    let graph = network.graph();
    let mapper = network.mapper();
    assert_eq!(graph.outputs().len(), 3);

    let post_process = move |value: Value, tensor: Tensor| {
        match graph.outputs().iter().index_of(&value) {
            None => None,
            Some(0) => Some(tensor.mapv(f32::tanh).to_shared()),
            Some(1) => Some(softmax(tensor, Axis(1)).to_shared()),
            Some(2) => {
                let mut result_logit: Array2<f32> = tensor.reshape((boards.len(), mapper.policy_len())).to_owned();
                for (bi, board) in boards.iter().enumerate() {
                    for i in 0..mapper.policy_len() {
                        let is_available = mapper.index_to_move(board, i)
                            .map_or(false, |mv| board.is_available_move(mv));

                        if !is_available {
                            result_logit[(bi, i)] = f32::NEG_INFINITY;
                        }
                    }
                }
                let result = softmax(result_logit, Axis(1))
                    .into_shape([&[boards.len()][..], mapper.policy_shape()].concat())
                    .unwrap().to_shared();
                Some(result)
            }
            _ => unreachable!(),
        }
    };

    visualize_graph_activations(graph, &exec, post_process, max_images)
}
