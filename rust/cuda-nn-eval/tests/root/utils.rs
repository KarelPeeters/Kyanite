use itertools::Itertools;

use cuda_nn_eval::tester::{assert_outputs_match, CHECK_BATCH_SIZE, eval_cudnn, load_check_data};
use nn_graph::cpu::{cpu_execute_graph, Tensor};
use nn_graph::graph::{Graph, Value};
use nn_graph::ndarray::ArcArray;
use nn_graph::onnx::load_graph_from_onnx_bytes;
use nn_graph::shape::Shape;

pub fn test_all(graph: &Graph, batch_size: usize, inputs: &[Tensor], expected_outputs: &[Tensor]) {
    println!("Testing:\n{}", graph);

    println!("Testing with CPU");

    let cpu_inputs = inputs.iter().collect_vec();
    let cpu_outputs = cpu_execute_graph(graph, batch_size, &cpu_inputs).outputs();
    assert_outputs_match(expected_outputs, &cpu_outputs, true);

    println!("Testing with Cudnn");
    let gpu_outputs = eval_cudnn(graph, batch_size, inputs);
    assert_outputs_match(expected_outputs, &gpu_outputs, true)
}

pub fn test_elementwise_pair(op: impl Fn(f32, f32) -> f32, graph_op: impl Fn(&mut Graph, Value, Value) -> Value) {
    let mut graph = Graph::new();

    let values = vec![0.0, 1.0, 2.0, 5.0, 6.0, 7.0, -1.0, -1.0, 0.5, 100.0, -100.0];
    let pair_count = values.len() * values.len();

    let left = graph.input(Shape::fixed(&[pair_count]));
    let right = graph.input(Shape::fixed(&[pair_count]));

    let output = graph_op(&mut graph, left, right);
    graph.output(output);

    let left_tensor = ArcArray::from_shape_fn(pair_count, |i| {
        values[i / values.len()]
    }).into_dyn();
    let right_tensor = ArcArray::from_shape_fn(pair_count, |i| {
        values[i % values.len()]
    }).into_dyn();
    let expected_output = ArcArray::from_shape_fn(pair_count, |i| {
        op(values[i / values.len()], values[i % values.len()])
    }).into_dyn();

    test_all(&graph, 0, &[left_tensor, right_tensor], &[expected_output]);
}

pub fn test_elementwise(op: impl Fn(f32) -> f32, graph_op: impl Fn(&mut Graph, Value) -> Value) {
    test_elementwise_pair(
        |left, _| op(left),
        |graph, left, _| graph_op(graph, left),
    )
}

pub fn test_onnx_bin(onnx: &[u8], bin: &[u8]) {
    let graph = load_graph_from_onnx_bytes(onnx);
    let (inputs, expected_outputs) = load_check_data(&graph, bin);
    test_all(&graph, CHECK_BATCH_SIZE, &inputs, &expected_outputs);
}
