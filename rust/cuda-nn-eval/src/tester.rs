use bytemuck::cast_slice_mut;
use itertools::{Itertools, zip_eq};

use cuda_sys::wrapper::handle::Device;
use nn_graph::cpu::Tensor;
use nn_graph::graph::{Graph, Value};
use nn_graph::ndarray::{Dimension, IxDyn};

use crate::executor::CudnnExecutor;

/// Check that the given graph produces the correct outputs as described by `check_data`,
/// which typically comes from a `.bin` file next to the `.onnx` file.
pub fn check_cudnn(graph: &Graph, check_data_bytes: &[u8], use_graph: bool) {
    let (batch_size, inputs, expected_outputs) = load_check_data(graph, check_data_bytes);
    let outputs = eval_cudnn(graph, batch_size, &inputs, false, use_graph);
    assert_outputs_match(graph.outputs(), &expected_outputs, &outputs, false);
}

const ERROR_TOLERANCE: f32 = 0.001;

pub fn assert_outputs_match(output_values: &[Value], expected_outputs: &[Tensor], outputs: &[Tensor], print: bool) {
    assert_eq!(expected_outputs.len(), outputs.len(), "Wrong number of outputs");

    let mut max_error = 0.0;

    for (i, (expected_output, output)) in zip_eq(expected_outputs, outputs).enumerate() {
        assert_eq!(expected_output.shape(), output.shape(), "Wrong output shape for output {}", i);

        for ((indices, &expected_value), &value) in zip_eq(expected_output.indexed_iter(), output.iter()) {
            let error = (expected_value - value).abs();
            max_error = f32::max(max_error, error);
            assert!(
                error < ERROR_TOLERANCE,
                "Wrong output value {}, expected {} at indices {:?} in output {} (value {:?})",
                value, expected_value, indices.slice(), i, output_values[i],
            )
        }

        if print {
            println!("Output {} matched, max error {}", i, max_error);
        }
    }
}

pub fn eval_cudnn(graph: &Graph, batch_size: usize, inputs: &[Tensor], print: bool, use_graph: bool) -> Vec<Tensor> {
    let inputs = inputs.iter()
        .map(|x| x.as_slice().expect("Only sliceable inputs supported in test framework"))
        .collect_vec();

    let mut executor = CudnnExecutor::new(Device::new(0), graph, batch_size, use_graph);
    if print {
        println!("{:?}", executor);
    }

    let gpu_outputs = executor.evaluate(&inputs);

    // turn into Tensors, using the cpu shapes
    let outputs = zip_eq(graph.outputs(), gpu_outputs)
        .map(|(&value, output)| {
            let shape = graph[value].shape.eval(batch_size);
            Tensor::from_shape_vec(&*shape.dims, output.clone())
                .expect("GPU output has wrong length")
        })
        .collect_vec();

    outputs
}

/// Load the check data into `(batch_size, inputs, expected_outputs)`.
pub fn load_check_data(graph: &Graph, check_data_bytes: &[u8]) -> (usize, Vec<Tensor>, Vec<Tensor>) {
    assert!(!check_data_bytes.is_empty(), "Check data must have at least one byte, the batch size");
    let batch_size = check_data_bytes[0] as usize;

    assert_eq!(
        (check_data_bytes.len() - 1) % 4, 0,
        "Data byte count must be multiple of 4 + 1 to be able to cast to float, got {}",
        check_data_bytes.len()
    );

    // copy the data into a float array instead of just casting it to ensure it's properly aligned
    let mut check_data = vec![0.0; (check_data_bytes.len() - 1) / 4];
    cast_slice_mut(&mut check_data).copy_from_slice(&check_data_bytes[1..]);

    let mut buf = &*check_data;
    let inputs = load_check_values(graph, batch_size, &mut buf, graph.inputs());
    let expected_outputs = load_check_values(graph, batch_size, &mut buf, graph.outputs());

    assert!(buf.is_empty(), "Leftover elements in check data buffer: {}", buf.len());

    (batch_size, inputs, expected_outputs)
}

/// Load the given values from the buffer while advancing it.
fn load_check_values(graph: &Graph, batch_size: usize, buf: &mut &[f32], values: &[Value]) -> Vec<Tensor> {
    values.iter()
        .map(|&value| {
            let shape = graph[value].shape.eval(batch_size);
            let tensor = Tensor::from_shape_vec(
                IxDyn(&shape.dims),
                buf[0..shape.size()].to_vec(),
            ).unwrap();
            *buf = &buf[shape.size()..];
            tensor
        })
        .collect_vec()
}
