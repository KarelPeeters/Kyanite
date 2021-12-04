use nn_graph::graph::{Graph, Value};
use nn_graph::ndarray::s;
use nn_graph::shape;
use nn_graph::shape::{Shape, Size};

use crate::root::runner::test_all;
use crate::root::tensor_utils::{linspace_tensor, linspace_vec, manual_tensor, range_vec};

#[test]
fn empty() {
    test_all(&Graph::new(), 8, &[], Some(&[]))
}

#[test]
fn copy() {
    let mut graph = Graph::new();

    let fixed_size = 10;
    let batch_size = 8;

    let fixed = graph.input(shape![fixed_size]);
    let batch = graph.input(shape![Size::BATCH]);
    graph.output_all(&[fixed, batch]);

    let fixed_tensor = linspace_tensor(fixed_size).into_dyn();
    let batch_tensor = linspace_tensor(batch_size).into_dyn();

    test_all(
        &graph,
        batch_size,
        &[fixed_tensor.to_shared(), batch_tensor.to_shared()],
        Some(&[fixed_tensor, batch_tensor]),
    )
}

#[test]
fn slice() {
    let mut graph = Graph::new();

    let input = graph.input(shape![10, 4]);
    let indexed = graph.index(input, 1, 0);
    let sliced = graph.slice(input, 0, 0, 2);
    let both = graph.slice(indexed, 0, 0, 2);
    graph.output_all(&[indexed, sliced, both]);

    let input_tensor = linspace_tensor((10, 4));

    test_all(
        &graph,
        0,
        &[input_tensor.to_shared().into_dyn()],
        Some(&[
            input_tensor.slice(s![.., 0]).into_dyn().to_shared(),
            input_tensor.slice(s![0..2, ..]).into_dyn().to_shared(),
            input_tensor.slice(s![0..2, 0]).into_dyn().to_shared(),
        ]),
    )
}

#[test]
fn linear() {
    let mut graph = Graph::new();

    let input = graph.input(shape![1, 4]);
    let weight = graph.constant(shape![2, 4], range_vec(8));
    let bias = graph.constant(shape![1, 2], vec![-10.0, 10.0]);

    let linear = graph.linear(input, weight);
    let biased = graph.add(linear, bias);

    graph.output_all(&[linear, biased]);

    test_all(
        &graph,
        0,
        &[manual_tensor((1, 4), vec![0.0, 1.0, 2.0, 3.0])],
        Some(&[
            manual_tensor((1, 2), vec![14.0, 38.0]),
            manual_tensor((1, 2), vec![4.0, 48.0]),
        ]),
    )
}

#[test]
fn fuse_clamp() {
    let mut graph = Graph::new();

    let mut curr = graph.input(shape![Size::BATCH]);

    curr = graph.clamp(curr, -5.0, f32::INFINITY);
    curr = graph.clamp(curr, f32::NEG_INFINITY, 2.0);
    curr = graph.clamp(curr, 0.0, 1.0);
    curr = graph.clamp(curr, -1.0, 2.0);

    graph.output(curr);

    test_all(
        &graph,
        5,
        &[manual_tensor(5, vec![-2.0, 0.0, 0.5, 1.0, 2.0])],
        Some(&[manual_tensor(5, vec![0.0, 0.0, 0.5, 1.0, 1.0])]),
    )
}

#[test]
fn add_broadcast() {
    let mut graph = Graph::new();

    let left = graph.constant(shape![2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let right = graph.constant(shape![1, 2, 2], vec![0.0, 1.0, 2.0, 3.0]);
    let output = graph.add(left, right);
    graph.output(output);

    test_all(
        &graph,
        0,
        &[],
        Some(&[manual_tensor((2, 2, 2), vec![0.0, 2.0, 4.0, 6.0, 4.0, 6.0, 8.0, 10.0])]),
    )
}

#[test]
fn affine_single_element() {
    let input_data = manual_tensor((8, 1, 1, 1), range_vec(8));
    let output_data = input_data.map(|&x| ((x + 1.0) * 2.0 * 10.0 + 3.0) * 4.0).to_shared();

    let mut graph = Graph::new();

    let const_shape = shape![1, 1, 1, 1];
    let bias_0 = graph.constant(const_shape.clone(), vec![1.0]);
    let scale_0 = graph.constant(const_shape.clone(), vec![2.0]);
    let filter = graph.constant(const_shape.clone(), vec![10.0]);
    let bias_1 = graph.constant(const_shape.clone(), vec![3.0]);
    let scale_1 = graph.constant(const_shape.clone(), vec![4.0]);

    let curr = graph.input(Shape::fixed(input_data.shape()));
    let curr = graph.add(curr, bias_0);
    let curr = graph.mul(curr, scale_0);
    let curr = graph.conv(curr, filter, 0);
    let curr = graph.add(curr, bias_1);
    let curr = graph.mul(curr, scale_1);
    graph.output(curr);

    test_all(
        &graph,
        0,
        &[input_data],
        Some(&[output_data]),
    )
}

#[test]
fn affine_multiple_channels() {
    let input_data = manual_tensor((8, 3, 1, 1), range_vec(8 * 3));

    let mut graph = Graph::new();

    let before_shape = shape![1, 3, 1, 1];
    let after_shape = shape![1, 2, 1, 1];
    let filter_shape = shape![2, 3, 1, 1];

    let bias_0 = graph.constant(before_shape.clone(), vec![1.0, 2.0, 3.0]);
    let scale_0 = graph.constant(before_shape.clone(), vec![2.0, 3.0, 4.0]);
    let filter = graph.constant(filter_shape.clone(), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let bias_1 = graph.constant(after_shape.clone(), vec![3.0, 4.0]);
    let scale_1 = graph.constant(after_shape.clone(), vec![4.0, 5.0]);

    let curr = graph.input(Shape::fixed(input_data.shape()));
    let curr = graph.add(curr, bias_0);
    let curr = graph.mul(curr, scale_0);
    let curr = graph.conv(curr, filter, 0);
    let curr = graph.add(curr, bias_1);
    let curr = graph.mul(curr, scale_1);
    graph.output(curr);

    test_all(
        &graph,
        0,
        &[input_data],
        None,
    )
}

#[test]
fn affine_padding() {
    let input_data = linspace_tensor((8, 3, 8, 8)).into_dyn();
    let filter_data = linspace_tensor((5, 3, 3, 3));

    let mut graph = Graph::new();

    let filter = graph.constant(Shape::fixed(filter_data.shape()), filter_data.to_owned().into_raw_vec());
    let bias_0 = graph.constant(shape![1, 3, 1, 1], linspace_vec(3));
    let bias_1 = graph.constant(shape![1, 5, 1, 1], linspace_vec(5));

    let mut curr = graph.input(Shape::fixed(&input_data.shape()));
    curr = graph.add(curr, bias_0);
    curr = graph.conv(curr, filter, 1);
    curr = graph.add(curr, bias_1);
    graph.output(curr);

    test_all(
        &graph,
        0,
        &[input_data],
        None,
    )
}

#[test]
fn pre_act_resnet() {
    let mut graph = Graph::new();

    let input_data = linspace_tensor((8, 3, 8, 8)).into_dyn();
    let input = graph.input(Shape::fixed(input_data.shape()));

    let filter_initial = graph.constant(shape![5, 3, 3, 3], linspace_vec(5 * 3 * 3 * 3));
    let filter_tower = graph.constant(shape![5, 5, 3, 3], linspace_vec(5 * 5 * 3 * 3));
    let filter_policy = graph.constant(shape![2, 5, 1, 1], linspace_vec(5 * 2));

    let mut tower = graph.conv(input, filter_initial, 1);
    for _ in 0..4 {
        let mut curr = channel_batchnorm(&mut graph, tower);
        curr = graph.clamp(curr, 0.0, 6.0);
        curr = graph.conv(curr, filter_tower, 1);
        curr = graph.clamp(curr, 0.0, 6.0);
        curr = graph.conv(curr, filter_tower, 1);
        tower = graph.add(curr, tower);
    }

    let policy = graph.conv(tower, filter_policy, 0);

    graph.output(tower);
    graph.output(policy);

    test_all(
        &graph,
        0,
        &[input_data],
        None,
    )
}

fn channel_batchnorm(graph: &mut Graph, input: Value) -> Value {
    let [_, c, _, _] = graph[input].shape.unwrap_4();
    let c = c.unwrap_fixed("Dummy BN channel count");

    let const_shape = shape![1, c, 1, 1];

    let mean = graph.constant(const_shape.clone(), linspace_vec(c));
    let var = graph.constant(const_shape.clone(), linspace_vec(c));
    let scale = graph.constant(const_shape.clone(), linspace_vec(c));
    let bias = graph.constant(const_shape.clone(), linspace_vec(c));

    let mut curr = input;
    curr = graph.add(curr, mean);
    curr = graph.mul(curr, var);
    curr = graph.mul(curr, scale);
    curr = graph.add(curr, bias);
    curr
}

#[test]
fn fuse_res() {
    let mut graph = Graph::new();

    let input = graph.input(shape![10, 4, 8, 8]);
    let other = graph.input(shape![10, 4, 8, 8]);
    let filter = graph.constant(shape![4, 4, 3, 3], linspace_vec(4 * 4 * 3 * 3));

    let mut curr = input;
    curr = graph.conv(curr, filter, 1);
    curr = graph.add(curr, other);
    curr = graph.clamp(curr, 0.0, f32::INFINITY);
    graph.output(curr);

    test_all(
        &graph, 0,
        &[
            linspace_tensor((10, 4, 8, 8)).into_dyn(),
            linspace_tensor((10, 4, 8, 8)).into_dyn() + 1.0
        ],
        None,
    );
}