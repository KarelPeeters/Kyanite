use bytemuck::cast_slice_mut;
use itertools::{enumerate, zip_eq, Itertools};

use cuda_sys::wrapper::handle::Device;
use nn_graph::cpu::Tensor;
use nn_graph::graph::{Graph, Value};
use nn_graph::ndarray::{Dimension, IxDyn};

use crate::executor::CudaExecutor;

/// Check that the given graph produces the correct outputs as described by `check_data`,
/// which typically comes from a `.bin` file next to the `.onnx` file.
pub fn check_cudnn(graph: &Graph, check_data_bytes: &[u8]) {
    let (batch_size, inputs, expected_outputs) = load_check_data(graph, check_data_bytes);
    let outputs = eval_cudnn(graph, batch_size, &inputs, true);
    assert_tensors_match(&expected_outputs, &outputs, true);
}

pub fn eval_cudnn(graph: &Graph, batch_size: usize, inputs: &[Tensor], print_executor: bool) -> Vec<Tensor> {
    let mut executor = CudaExecutor::new(Device::new(0), graph, batch_size);
    if print_executor {
        println!("{:?}", executor);
    }
    executor.evaluate_tensors(inputs)
}

const TOLERANCE_ABS_DIFF: f32 = 0.001;
const TOLERANCE_REL_DIFF: f32 = 0.001;
const MAX_LOGGED_ERRORS: usize = 8;

pub fn assert_tensors_match(expected: &[Tensor], actual: &[Tensor], print_match: bool) {
    match check_tensors_match(expected, actual) {
        Ok(Match {
            diff_per_tensor: diff_per_output,
        }) => {
            if print_match {
                for (i, diff) in enumerate(diff_per_output) {
                    let Difference {
                        max_abs_diff,
                        max_rel_diff,
                    } = diff;

                    println!(
                        "Output {} with shape {:?} matched, max diff: abs {}, rel {}",
                        i,
                        actual[i].shape(),
                        max_abs_diff,
                        max_rel_diff
                    );
                }
            }
        }
        Err(Mismatch {
            error_count,
            total_count,
            first_errors,
        }) => {
            eprintln!("Mismatch in {}/{} values:", error_count, total_count);

            for error in &first_errors {
                let Error {
                    tensor,
                    ref indices,
                    expected_value,
                    actual_value,
                } = *error;

                eprintln!(
                    "  Wrong output value {}, expected {} at indices {:?} in tensor {} (shape {:?})",
                    actual_value,
                    expected_value,
                    indices,
                    tensor,
                    expected[tensor].shape()
                )
            }

            if error_count > first_errors.len() {
                eprintln!("  ...");
            }

            panic!("Output mismatch");
        }
    }
}

#[derive(Debug, Clone)]
pub struct Match {
    pub diff_per_tensor: Vec<Difference>,
}

#[derive(Debug, Copy, Clone)]
pub struct Difference {
    pub max_rel_diff: f32,
    pub max_abs_diff: f32,
}

#[derive(Debug, Clone)]
pub struct Mismatch {
    pub error_count: usize,
    pub total_count: usize,
    pub first_errors: Vec<Error>,
}

#[derive(Debug, Clone)]
pub struct Error {
    pub tensor: usize,
    pub indices: Vec<usize>,
    pub expected_value: f32,
    pub actual_value: f32,
}

pub fn check_tensors_match(expected: &[Tensor], actual: &[Tensor]) -> Result<Match, Mismatch> {
    assert_eq!(expected.len(), actual.len(), "Wrong number of tensors");

    let mut total_error_count = 0;
    let mut total_element_count = 0;

    let mut diff_per_tensor = vec![];
    let mut first_errors = vec![];

    for (i, (expected_output, output)) in zip_eq(expected, actual).enumerate() {
        let mut current_error_count = 0;

        assert_eq!(
            expected_output.shape(),
            output.shape(),
            "Wrong output shape for tensor {}",
            i
        );

        let mut max_abs_diff = 0.0;
        let mut max_rel_diff = 0.0;

        for ((indices, &expected_value), &value) in zip_eq(expected_output.indexed_iter(), output.iter()) {
            let (abs_diff, rel_diff) = if expected_value == value || (expected_value.is_nan() && value.is_nan()) {
                (0.0, 0.0)
            } else {
                let abs_diff = (expected_value - value).abs();
                let rel_diff = abs_diff / expected_value.abs();
                (abs_diff, rel_diff)
            };

            max_abs_diff = f32::max(max_abs_diff, abs_diff);
            max_rel_diff = f32::max(max_rel_diff, rel_diff);

            total_element_count += 1;

            if abs_diff >= TOLERANCE_ABS_DIFF && rel_diff >= TOLERANCE_REL_DIFF {
                total_error_count += 1;
                current_error_count += 1;

                if current_error_count < MAX_LOGGED_ERRORS {
                    first_errors.push(Error {
                        tensor: i,
                        indices: indices.slice().to_vec(),
                        expected_value,
                        actual_value: value,
                    });
                }
            }
        }

        diff_per_tensor.push(Difference {
            max_rel_diff,
            max_abs_diff,
        });
    }

    if total_error_count == 0 {
        Ok(Match { diff_per_tensor })
    } else {
        Err(Mismatch {
            error_count: total_error_count,
            total_count: total_element_count,
            first_errors,
        })
    }
}

/// Load the check data into `(batch_size, inputs, expected_outputs)`.
pub fn load_check_data(graph: &Graph, check_data_bytes: &[u8]) -> (usize, Vec<Tensor>, Vec<Tensor>) {
    assert!(
        !check_data_bytes.is_empty(),
        "Check data must have at least one byte, the batch size"
    );
    let batch_size = check_data_bytes[0] as usize;

    assert_eq!(
        (check_data_bytes.len() - 1) % 4,
        0,
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
    values
        .iter()
        .map(|&value| {
            let shape = graph[value].shape.eval(batch_size);
            let tensor = Tensor::from_shape_vec(IxDyn(&shape.dims), buf[0..shape.size()].to_vec()).unwrap();
            *buf = &buf[shape.size()..];
            tensor
        })
        .collect_vec()
}
