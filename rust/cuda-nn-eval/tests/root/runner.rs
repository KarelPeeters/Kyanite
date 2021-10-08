use bytemuck::cast_slice;
use itertools::{Itertools, zip_eq};

use cuda_nn_eval::executor::CudnnExecutor;
use cuda_sys::wrapper::handle::CudnnHandle;
use nn_graph::cpu::{cpu_execute_graph, Tensor};
use nn_graph::graph::{Graph, Value};
use nn_graph::ndarray::{ArcArray, Dimension, IxDyn};
use nn_graph::onnx::load_graph_from_onnx_bytes;
use nn_graph::shape::Shape;

pub fn test_all(graph: &Graph, batch_size: usize, inputs: &[Tensor], expected_outputs: &[Tensor]) {
    println!("Testing:\n{:?}", graph);

    println!("Testing with CPU");
    let cpu_inputs = inputs.iter()
        .map(|x| x)
        .collect_vec();

    let cpu_outputs = cpu_execute_graph(graph, batch_size, &cpu_inputs).outputs();
    assert_outputs_match(expected_outputs, &cpu_outputs);

    println!("Testing with Cudnn");
    let gpu_inputs = inputs.iter()
        .map(|x| x.as_slice().expect("Only sliceable inputs supported in test framework"))
        .collect_vec();

    let mut executor = CudnnExecutor::new(CudnnHandle::new(0), graph, batch_size);
    let gpu_outputs = executor.evaluate(&gpu_inputs);

    // turn into Tensors, using the cpu shapes
    let gpu_outputs = zip_eq(cpu_outputs, gpu_outputs).into_iter()
        .map(|(cpu_x, gpu_x)| {
            Tensor::from_shape_vec(cpu_x.shape(), gpu_x.clone())
                .expect("GPU output has wrong length")
        })
        .collect_vec();
    assert_outputs_match(expected_outputs, &gpu_outputs)
}

const ERROR_TOLERANCE: f32 = 0.0001;

fn assert_outputs_match(expected_outputs: &[Tensor], outputs: &[Tensor]) {
    assert_eq!(expected_outputs.len(), outputs.len(), "Wrong number of outputs");

    let mut max_error = 0.0;

    for (i, (expected_output, output)) in zip_eq(expected_outputs, outputs).enumerate() {
        assert_eq!(expected_output.shape(), output.shape(), "Wrong output shape for output {}", i);

        for ((indices, &expected_value), &value) in zip_eq(expected_output.indexed_iter(), output.iter()) {
            let error = (expected_value - value).abs();
            max_error = f32::max(max_error, error);
            assert!(
                error < ERROR_TOLERANCE,
                "Wrong output value {}, expected {} at indices {:?} in output {}",
                value, expected_value, indices.slice(), i,
            )
        }

        println!("Output {} matched, max error {}", i, max_error);
    }
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

const CHECK_BATCH_SIZE: usize = 2;

pub struct OnnxPair {
    pub onnx: &'static [u8],
    pub bin: &'static [u8],
}

pub fn test_onnx_pair(pair: &OnnxPair) {
    let graph = load_graph_from_onnx_bytes(pair.onnx);

    let mut data_left: &[f32] = cast_slice(pair.bin);
    let inputs = load_values(&graph, &mut data_left, graph.inputs());
    let expected_outputs = load_values(&graph, &mut data_left, graph.outputs());
    assert_eq!(0, data_left.len(), "Leftover data");

    test_all(&graph, CHECK_BATCH_SIZE, &inputs, &expected_outputs);
}

/// Load the given values from the buffer while advancing it.
fn load_values(graph: &Graph, buf: &mut &[f32], values: &[Value]) -> Vec<Tensor> {
    values.iter()
        .map(|&value| {
            let shape = graph[value].shape.eval(CHECK_BATCH_SIZE);
            let tensor = Tensor::from_shape_vec(
                IxDyn(&shape.dims),
                buf[0..shape.size()].to_vec(),
            ).unwrap();
            *buf = &buf[shape.size()..];
            tensor
        })
        .collect_vec()
}