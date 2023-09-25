use kn_cuda_eval::tester::{assert_tensors_match, eval_cudnn, load_check_data};
use kn_graph::cpu::{cpu_eval_graph, Tensor};
use kn_graph::graph::{Graph, Value};
use kn_graph::ndarray::ArcArray;
use kn_graph::onnx::load_graph_from_onnx_bytes;
use kn_graph::optimizer::{optimize_graph, OptimizerSettings};
use kn_graph::shape;

pub fn test_all(graph: &Graph, batch_size: usize, inputs: &[Tensor], expected_outputs: Option<&[Tensor]>) {
    if expected_outputs.is_none() {
        println!("No expected outputs provided, using unoptimized cpu outputs instead");
    }

    println!("Running unoptimized CPU");

    println!("Testing unoptimized");
    let cpu_outputs = test_all_exact_graph(&graph, batch_size, inputs, expected_outputs);
    let expected_outputs = expected_outputs.unwrap_or(&cpu_outputs);

    println!("Optimizing graph");
    let optimized = optimize_graph(&graph, OptimizerSettings::default());

    println!("Testing optimized");
    test_all_exact_graph(&optimized, batch_size, inputs, Some(expected_outputs));
}

// TODO remove references to this
pub fn test_all_exact_graph(
    graph: &Graph,
    batch_size: usize,
    inputs: &[Tensor],
    expected_outputs: Option<&[Tensor]>,
) -> Vec<Tensor> {
    println!("Testing:\n{}", graph);

    println!("Testing CPU");

    let cpu_outputs = cpu_eval_graph(graph, batch_size, inputs);

    let expected_outputs = if let Some(expected_outputs) = expected_outputs {
        assert_tensors_match(expected_outputs, &cpu_outputs, true);
        expected_outputs
    } else {
        &cpu_outputs
    };
    println!();

    println!("Testing GPU");
    let gpu_outputs = eval_cudnn(graph, batch_size, inputs, true);
    assert_tensors_match(expected_outputs, &gpu_outputs, true);
    println!();

    cpu_outputs
}

pub const ELEMENTWISE_TEST_VALUES: &[f32] = &[
    0.0, 0.1, 0.5, 0.8, 1.0, 1.1, 1.5, 1.7, 2.0, 5.0, 6.0, 7.0, -1.0, -1.0, 0.5, 20.0, 80.0, 100.0, 1000.0, -20.0,
    -100.0,
];

pub fn test_elementwise_pair(op: impl Fn(f32, f32) -> f32, graph_op: impl Fn(&mut Graph, Value, Value) -> Value) {
    let mut graph = Graph::new();

    let values = ELEMENTWISE_TEST_VALUES;
    let pair_count = values.len() * values.len();

    let left = graph.input(shape![pair_count]);
    let right = graph.input(shape![pair_count]);

    let output = graph_op(&mut graph, left, right);
    graph.output(output);

    let left_tensor = ArcArray::from_shape_fn(pair_count, |i| values[i / values.len()]).into_dyn();
    let right_tensor = ArcArray::from_shape_fn(pair_count, |i| values[i % values.len()]).into_dyn();
    let expected_output =
        ArcArray::from_shape_fn(pair_count, |i| op(values[i / values.len()], values[i % values.len()])).into_dyn();

    test_all(&graph, 0, &[left_tensor, right_tensor], Some(&[expected_output]));
}

pub fn test_elementwise(op: impl Fn(f32) -> f32, graph_op: impl Fn(&mut Graph, Value) -> Value) {
    let mut graph = Graph::new();

    let values = ELEMENTWISE_TEST_VALUES;

    let input = graph.input(shape![values.len()]);
    let output = graph_op(&mut graph, input);
    graph.output(output);

    let input_tensor = ArcArray::from_shape_vec(values.len(), values.to_vec())
        .unwrap()
        .into_dyn();
    let expected_output = input_tensor.map(|&v| op(v)).into_shared().into_dyn();

    test_all(&graph, 0, &[input_tensor], Some(&[expected_output]));
}

pub fn test_onnx_bin(onnx: &[u8], bin: &[u8]) {
    let graph = load_graph_from_onnx_bytes(onnx).unwrap();
    let (batch_size, inputs, expected_outputs) = load_check_data(&graph, bin);
    println!("Loaded batch size {}", batch_size);
    test_all(&graph, batch_size, &inputs, Some(&expected_outputs));
}
