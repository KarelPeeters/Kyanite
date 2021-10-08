use itertools::Itertools;

use nn_graph::cpu::Tensor;
use nn_graph::graph::Graph;
use nn_graph::ndarray::{ArcArray, Dimension, IntoDimension};
use nn_graph::ndarray::s;
use nn_graph::shape::{Shape, Size};

use crate::root::utils::test_all;

#[test]
fn empty() {
    test_all(&Graph::new(), 8, &[], &[])
}

#[test]
fn copy() {
    let mut graph = Graph::new();

    let fixed_size = 10;
    let batch_size = 8;

    let fixed = graph.input(Shape::fixed(&[fixed_size]));
    let batch = graph.input(Shape::new(vec![Size::BATCH]));
    graph.output_all(&[fixed, batch]);

    let fixed_tensor = linspace_tensor(fixed_size).into_dyn();
    let batch_tensor = linspace_tensor(batch_size).into_dyn();

    test_all(
        &graph,
        batch_size,
        &[fixed_tensor.to_shared(), batch_tensor.to_shared()],
        &[fixed_tensor, batch_tensor],
    )
}

#[test]
fn slice() {
    let mut graph = Graph::new();

    let input = graph.input(Shape::fixed(&[10, 4]));
    let indexed = graph.index(input, 1, 0);
    let sliced = graph.slice(input, 0, 0, 2);
    let both = graph.slice(indexed, 0, 0, 2);
    graph.output_all(&[indexed, sliced, both]);

    let input_tensor = linspace_tensor((10, 4));

    test_all(
        &graph,
        0,
        &[input_tensor.to_shared().into_dyn()],
        &[
            input_tensor.slice(s![.., 0]).into_dyn().to_shared(),
            input_tensor.slice(s![0..2, ..]).into_dyn().to_shared(),
            input_tensor.slice(s![0..2, 0]).into_dyn().to_shared(),
        ],
    )
}

#[test]
fn linear() {
    let mut graph = Graph::new();

    let input = graph.input(Shape::fixed(&[1, 4]));
    let weight = graph.constant(Shape::fixed(&[2, 4]), range_vec(8));
    let bias = graph.constant(Shape::fixed(&[1, 2]), vec![-10.0, 10.0]);

    let linear = graph.linear(input, weight);
    let biased = graph.add(linear, bias);

    graph.output_all(&[linear, biased]);

    test_all(
        &graph,
        0,
        &[manual_tensor((1, 4), vec![0.0, 1.0, 2.0, 3.0])],
        &[
            manual_tensor((1, 2), vec![14.0, 38.0]),
            manual_tensor((1, 2), vec![4.0, 48.0]),
        ],
    )
}

fn manual_tensor<I: IntoDimension>(shape: I, data: Vec<f32>) -> Tensor {
    ArcArray::from_shape_vec(shape, data)
        .expect("Shape and data length mismatch")
        .into_dyn()
}

fn linspace_tensor<I: IntoDimension + Copy>(shape: I) -> ArcArray<f32, I::Dim> {
    let size = shape.into_dimension().size();
    ArcArray::linspace(-5.0, 5.0, size)
        .reshape(shape)
}

fn range_vec(len: usize) -> Vec<f32> {
    (0..len).map(|x| x as f32).collect_vec()
}