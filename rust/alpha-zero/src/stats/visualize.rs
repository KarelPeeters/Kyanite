use board_game::board::{Board, Player};
use ndarray::{ArrayView1, s};

use nn_graph::cpu::{softmax, Tensor};
use nn_graph::graph::Value;
use nn_graph::ndarray::{Array2, Axis};
use nn_graph::visualize::{Image, VisTensor, visualize_graph_activations};

use crate::mapping::BoardMapper;
use crate::network::common::softmax_in_place;
use crate::network::cpu::CPUNetwork;
use crate::util::IndexOf;

pub fn visualize_network_activations_split<B: Board, M: BoardMapper<B>>(
    network: &mut CPUNetwork<B, M>,
    boards: &[B],
    max_images: Option<usize>,
    show_variance: bool,
) -> (Vec<Image>, Vec<Image>) {
    let (boards_a, boards_b) = split_player(boards);

    (
        visualize_network_activations(network, &boards_a, max_images, show_variance),
        visualize_network_activations(network, &boards_b, max_images, show_variance),
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
    show_variance: bool,
) -> Vec<Image> {
    let exec = network.evaluate_batch_exec(boards);

    let graph = network.graph();
    let mapper = network.mapper();
    assert_eq!(graph.outputs().len(), 3);

    let post_process = move |value: Value, tensor: Tensor| {
        match graph.outputs().iter().index_of(&value) {
            None => None,
            // tanh(value_logit), mapped to [0..1]
            Some(0) => Some(VisTensor::abs(tensor.mapv(|x| (x.tanh() + 1.0 / 2.0)).to_shared())),
            // softmax(wdl_logits)
            Some(1) => Some(VisTensor::abs(softmax(tensor, Axis(1)).to_shared())),
            // softmax(available_mask * policy_logits)
            Some(2) => {
                let flat_shape = (boards.len(), mapper.policy_len());
                let full_shape = [&[boards.len()][..], mapper.policy_shape()].concat();

                let flat_logits = tensor.reshape(flat_shape);
                let mut flat_result: Array2<f32> = Array2::zeros(flat_shape);

                let mut buffer = vec![];

                for (bi, board) in boards.iter().enumerate() {
                    let mut any_available = false;
                    buffer.clear();

                    for i in 0..mapper.policy_len() {
                        let is_available = mapper.index_to_move(board, i)
                            .map_or(false, |mv| board.is_available_move(mv));

                        any_available |= is_available;
                        if is_available {
                            buffer.push(flat_logits[(bi, i)]);
                        } else {
                            buffer.push(f32::NEG_INFINITY);
                        }
                    }

                    if any_available {
                        softmax_in_place(&mut buffer);
                        flat_result.slice_mut(s![bi, ..]).assign(&ArrayView1::from(&buffer));
                    }
                    // otherwise leave result as zero
                }
                let result = flat_result.into_shape(full_shape).unwrap().to_shared();
                Some(VisTensor::abs(result))
            }
            _ => unreachable!(),
        }
    };

    visualize_graph_activations(graph, &exec, post_process, max_images, show_variance)
}
