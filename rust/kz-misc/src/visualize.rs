use board_game::board::{Board, Player};
use ndarray::{s, ArrayView1, Slice};

use kz_core::mapping::BoardMapper;
use kz_core::network::common::softmax_in_place;
use kz_core::network::cpu::CPUNetwork;
use nn_graph::cpu::{softmax, Tensor};
use nn_graph::graph::Value;
use nn_graph::ndarray::{Array2, Axis};
use nn_graph::shape;
use nn_graph::shape::Size;
use nn_graph::visualize::{visualize_graph_activations, Image, VisTensor};

pub fn visualize_network_activations_split<B: Board, M: BoardMapper<B>>(
    network: &mut CPUNetwork<B, M>,
    boards: &[B],
    max_images: Option<usize>,
    show_variance: bool,
    print_details: bool,
) -> (Vec<Image>, Vec<Image>) {
    let (boards_a, boards_b) = split_player(boards);

    (
        // (only print the details once)
        visualize_network_activations(network, &boards_a, max_images, show_variance, print_details),
        visualize_network_activations(network, &boards_b, max_images, show_variance, false),
    )
}

fn split_player<B: Board>(boards: &[B]) -> (Vec<B>, Vec<B>) {
    let mut result_a = vec![];
    let mut result_b = vec![];

    for board in boards {
        match board.next_player() {
            Player::A => result_a.push(board.clone()),
            Player::B => result_b.push(board.clone()),
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
    print_details: bool,
) -> Vec<Image> {
    let exec = network.evaluate_batch_exec(boards, true);

    let graph = network.graph();
    let mapper = network.mapper();

    let output_count = graph.outputs().len();
    assert!(
        output_count == 2 || output_count == 3,
        "Expected either (value, wdl, policy) or (scalars, policy) as output, got {}",
        output_count
    );

    let post_process = move |value: Value, tensor: Tensor| {
        if !graph.outputs().contains(&value) {
            return vec![];
        }

        let value_shape = shape![Size::BATCH];
        let wdl_shape = shape![Size::BATCH, 3];
        let scalar_shape = shape![Size::BATCH, 5];
        let policy_shape = shape![Size::BATCH, mapper.policy_len()];

        let shape = &graph[value].shape;

        if shape == &value_shape {
            // tanh(value_logit) -> 0..1
            vec![VisTensor::abs(tensor.mapv(|x| (x.tanh() + 1.0 / 2.0)).to_shared())]
        } else if shape == &wdl_shape {
            // softmax(wdl_logits)
            vec![VisTensor::abs(softmax(tensor, Axis(1)).to_shared())]
        } else if shape == &scalar_shape {
            // tanh(value_logit) -> 0..1, softmax(wdl_logits), moves_left
            vec![
                VisTensor::abs(
                    tensor
                        .index_axis(Axis(1), 0)
                        .mapv(|x| (x.tanh() + 1.0 / 2.0))
                        .to_shared(),
                ),
                VisTensor::abs(softmax(tensor.slice_axis(Axis(1), Slice::from(1..4)), Axis(1)).to_shared()),
                VisTensor::norm(tensor.index_axis(Axis(1), 4).to_shared()),
            ]
        } else if shape == &policy_shape {
            let flat_shape = (boards.len(), mapper.policy_len());
            let full_shape = [&[boards.len()][..], mapper.policy_shape()].concat();

            let flat_logits = tensor.reshape(flat_shape);
            let mut flat_result: Array2<f32> = Array2::zeros(flat_shape);

            let mut buffer = vec![];

            for (bi, board) in boards.iter().enumerate() {
                let mut any_available = false;
                buffer.clear();

                for i in 0..mapper.policy_len() {
                    let is_available = mapper
                        .index_to_move(board, i)
                        .map_or(false, |mv| board.is_available_move(mv).unwrap_or(false));

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
            vec![VisTensor::abs(result)]
        } else {
            panic!("Unexpected output shape {}", shape);
        }
    };

    visualize_graph_activations(graph, &exec, post_process, max_images, show_variance, print_details)
}
