use bytemuck::cast_slice_mut;
use itertools::{enumerate, zip_eq, Itertools};

use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::dispatch_dtensor_pair;
use kn_graph::dtype::{DScalar, DTensor, DType, IntoDScalar, Tensor};
use kn_graph::graph::{Graph, Value};
use kn_graph::ndarray::{Dimension, IxDyn};

use crate::executor::CudaExecutor;

/// Check that the given graph produces the correct outputs as described by `check_data`,
/// which typically comes from a `.bin` file next to the `.onnx` file.
pub fn check_cudnn(graph: &Graph, check_data_bytes: &[u8]) {
    let (batch_size, inputs, expected_outputs) = load_check_data(graph, check_data_bytes);
    let outputs = eval_cudnn(graph, batch_size, &inputs, true);
    assert_tensors_match(&expected_outputs, &outputs, true);
}

pub fn eval_cudnn(graph: &Graph, batch_size: usize, inputs: &[DTensor], print_executor: bool) -> Vec<DTensor> {
    let mut executor = CudaExecutor::new(Device::new(0), graph, batch_size);
    if print_executor {
        println!("{:?}", executor);
    }
    executor.evaluate(inputs).to_owned()
}

const TOLERANCE_ABS_DIFF: f64 = 0.001;
const TOLERANCE_REL_DIFF: f64 = 0.001;
const MAX_LOGGED_ERRORS: usize = 8;

pub fn assert_tensors_match(expected: &[DTensor], actual: &[DTensor], print_match: bool) {
    match check_tensors_match(expected, actual) {
        Ok(Match {
            diff_per_tensor: diff_per_output,
        }) => {
            if print_match {
                for (i, diff) in enumerate(diff_per_output) {
                    match diff {
                        Difference::Float(DifferenceFloat {
                            max_abs_diff,
                            max_rel_diff,
                        }) => {
                            println!(
                                "Output {} with shape {:?} and {:?} matched, max diff: abs {}, rel {}",
                                i,
                                actual[i].shape(),
                                actual[i].dtype(),
                                max_abs_diff,
                                max_rel_diff
                            );
                        }
                        Difference::IntMatch => {
                            println!(
                                "Output {} with shape {:?} and {:?} matched",
                                i,
                                actual[i].shape(),
                                actual[i].dtype(),
                            );
                        }
                    }
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
                    more_omitted,
                } = *error;

                eprintln!(
                    "  Wrong output value {:?}, expected {:?} at indices {:?} in tensor {} (shape {:?})",
                    actual_value,
                    expected_value,
                    indices,
                    tensor,
                    expected[tensor].shape()
                );

                if more_omitted {
                    eprintln!("  ...");
                }
            }

            panic!("Output mismatch");
        }
    }
}

#[derive(Debug, Clone)]
pub struct Match {
    pub diff_per_tensor: Vec<Difference>,
}

#[derive(Debug, Clone)]
pub enum Difference {
    Float(DifferenceFloat),
    IntMatch,
}

// TODO int/float enum? or just float/dscalar?
#[derive(Debug, Copy, Clone)]
pub struct DifferenceFloat {
    pub max_rel_diff: f64,
    pub max_abs_diff: f64,
}

#[derive(Debug, Clone)]
pub struct Mismatch {
    pub error_count: u64,
    pub total_count: u64,
    pub first_errors: Vec<Error>,
}

#[derive(Debug, Clone)]
pub struct Error {
    pub tensor: usize,
    pub indices: Vec<usize>,
    pub expected_value: DScalar,
    pub actual_value: DScalar,
    pub more_omitted: bool,
}

#[derive(Default, Debug, Clone)]
pub struct Counts {
    total_element_count: u64,
    total_error_count: u64,
}

pub fn check_tensors_match(expected: &[DTensor], actual: &[DTensor]) -> Result<Match, Mismatch> {
    assert_eq!(expected.len(), actual.len(), "Wrong number of tensors");

    let mut counts = Counts::default();

    let mut diff_per_tensor = vec![];
    let mut first_errors = vec![];

    for (i, (expected_output, output)) in zip_eq(expected, actual).enumerate() {
        let diff = check_tensor_match(i, expected_output, output, &mut counts, &mut first_errors);
        diff_per_tensor.push(diff);
    }

    if counts.total_error_count == 0 {
        Ok(Match { diff_per_tensor })
    } else {
        Err(Mismatch {
            error_count: counts.total_error_count,
            total_count: counts.total_element_count,
            first_errors,
        })
    }
}

fn check_tensor_match(
    i: usize,
    expected_output: &DTensor,
    output: &DTensor,
    counts: &mut Counts,
    first_errors: &mut Vec<Error>,
) -> Difference {
    assert_eq!(
        expected_output.shape(),
        output.shape(),
        "Wrong output shape for tensor {}",
        i
    );
    assert_eq!(
        expected_output.dtype(),
        output.dtype(),
        "Wrong output dtype for tensor {}",
        i
    );
    let dtype = expected_output.dtype();

    match dtype {
        DType::F32 => Difference::Float(check_tensor_match_approx(
            i,
            &expected_output.unwrap_f32().unwrap().mapv(|x| x as f64).into_shared(),
            &output.unwrap_f32().unwrap().mapv(|x| x as f64).into_shared(),
            counts,
            first_errors,
        )),
        DType::F64 => Difference::Float(check_tensor_match_approx(
            i,
            expected_output.unwrap_f64().unwrap(),
            output.unwrap_f64().unwrap(),
            counts,
            first_errors,
        )),
        DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
            dispatch_dtensor_pair!(expected_output, output, |_T, _f, expected_output, output| {
                check_tensor_match_exact(i, expected_output, output, counts, first_errors)
            })
        }
    }
}

fn check_tensor_match_exact<T: IntoDScalar>(
    i: usize,
    expected_output: &Tensor<T>,
    output: &Tensor<T>,
    counts: &mut Counts,
    first_errors: &mut Vec<Error>,
) -> Difference {
    assert!(T::DTYPE.is_int());

    let mut current_error_count = 0;

    for ((indices, &expected_value), &value) in zip_eq(expected_output.indexed_iter(), output.iter()) {
        counts.total_element_count += 1;

        if expected_value != value {
            counts.total_error_count += 1;
            current_error_count += 1;

            if current_error_count < MAX_LOGGED_ERRORS {
                first_errors.push(Error {
                    tensor: i,
                    indices: indices.slice().to_vec(),
                    expected_value: expected_value.to_dscalar(),
                    actual_value: value.to_dscalar(),
                    more_omitted: false,
                });
            } else {
                first_errors.last_mut().unwrap().more_omitted = true;
            }
        }
    }

    Difference::IntMatch
}

fn check_tensor_match_approx(
    i: usize,
    expected_output: &Tensor<f64>,
    output: &Tensor<f64>,
    counts: &mut Counts,
    first_errors: &mut Vec<Error>,
) -> DifferenceFloat {
    let mut max_abs_diff = 0.0;
    let mut max_rel_diff = 0.0;

    let mut current_error_count = 0;

    for ((indices, &expected_value), &value) in zip_eq(expected_output.indexed_iter(), output.iter()) {
        let (abs_diff, rel_diff) = if expected_value == value || expected_value.is_nan() || value.is_nan() {
            (0.0, 0.0)
        } else {
            let abs_diff = (expected_value - value).abs();
            let rel_diff = abs_diff / expected_value.abs();
            (abs_diff, rel_diff)
        };

        max_abs_diff = f64::max(max_abs_diff, abs_diff);
        max_rel_diff = f64::max(max_rel_diff, rel_diff);

        counts.total_element_count += 1;

        let exceeds_tolerance = abs_diff >= TOLERANCE_ABS_DIFF && rel_diff >= TOLERANCE_REL_DIFF;
        let nan_mismatch = expected_value.is_nan() != value.is_nan();

        if exceeds_tolerance || nan_mismatch {
            counts.total_error_count += 1;
            current_error_count += 1;

            if current_error_count < MAX_LOGGED_ERRORS {
                first_errors.push(Error {
                    tensor: i,
                    indices: indices.slice().to_vec(),
                    expected_value: expected_value.to_dscalar(),
                    actual_value: value.to_dscalar(),
                    more_omitted: false,
                });
            } else {
                first_errors.last_mut().unwrap().more_omitted = true;
            }
        }
    }

    DifferenceFloat {
        max_rel_diff,
        max_abs_diff,
    }
}

/// Load the check data into `(batch_size, inputs, expected_outputs)`.
pub fn load_check_data(graph: &Graph, check_data_bytes: &[u8]) -> (usize, Vec<DTensor>, Vec<DTensor>) {
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
fn load_check_values(graph: &Graph, batch_size: usize, buf: &mut &[f32], values: &[Value]) -> Vec<DTensor> {
    // TODO support loading non-f32 values
    values
        .iter()
        .map(|&value| {
            let shape = graph[value].shape.eval(batch_size);
            let tensor =
                DTensor::F32(Tensor::from_shape_vec(IxDyn(&shape.dims), buf[0..shape.size()].to_vec()).unwrap());
            *buf = &buf[shape.size()..];
            tensor
        })
        .collect_vec()
}
