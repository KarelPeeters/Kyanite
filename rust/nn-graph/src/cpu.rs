use std::fmt::{Debug, Display, Formatter};
use std::time::Instant;

use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{
    s, ArcArray, Array3, Array4, ArrayView, ArrayView3, ArrayView4, Ix3, IxDyn, SliceInfo, SliceInfoElem, Zip,
};

use crate::graph::{ConvDetails, Graph, Operation, Value, ValueInfo};
use crate::ndarray::{Array, ArrayBase, Axis};
use crate::shape::ConcreteShape;

/// We're using an ArcArray so reshaping is free.
pub type Tensor = ArcArray<f32, IxDyn>;

pub fn cpu_eval_graph(graph: &Graph, batch_size: usize, inputs: &[Tensor]) -> Vec<Tensor> {
    let exec = cpu_eval_graph_exec(&graph, batch_size, inputs, false);
    exec.output_tensors()
}

pub fn cpu_eval_graph_exec(graph: &Graph, batch_size: usize, inputs: &[Tensor], keep_all: bool) -> ExecutionInfo {
    assert_eq!(
        graph.inputs().len(),
        inputs.len(),
        "Wrong input count, graph has {} but got {}",
        graph.inputs().len(),
        inputs.len()
    );

    let mut map: IndexMap<Value, CalculatedValue> = IndexMap::default();

    for output in graph.values() {
        let info = &graph[output];

        let start_time = Instant::now();
        let result = run_cpu_operation(info, &map, inputs, batch_size);
        let end_time = Instant::now();

        let tensor_shape = ConcreteShape::new(result.shape().to_vec());

        let mut output_calc = CalculatedValue {
            value: output,
            tensor: Some(result),
            tensor_shape,
            uses_seen: 0,
            time_spent: (end_time - start_time).as_secs_f32(),
        };

        // free tensors that won't be used again
        if !keep_all {
            // immediately discard this output
            if graph.is_hidden_with_uses(output, 0) {
                output_calc.tensor = None
            }

            // discard inputs that just got used for the last time
            for input in graph[output].operation.inputs() {
                let input_calc: &mut CalculatedValue = map.get_mut(&input).unwrap();
                input_calc.uses_seen += 1;

                if graph.is_hidden_with_uses(input, input_calc.uses_seen) {
                    input_calc.tensor = None;
                }
            }
        }

        // store output for later
        let prev = map.insert(output, output_calc);
        assert!(prev.is_none());
    }

    ExecutionInfo {
        batch_size,
        values: map,
        outputs: graph.outputs().to_owned(),
    }
}

#[derive(Debug)]
pub enum OperationError {
    NoBatchSize,
    MissingOperand,
    MissingInput,
}

pub type OperationResult = Result<Tensor, OperationError>;

fn run_cpu_operation(
    info: &ValueInfo,
    map: &IndexMap<Value, CalculatedValue>,
    inputs: &[Tensor],
    batch_size: usize,
) -> Tensor {
    try_run_cpu_operation(
        info,
        |value| Ok(map.get(&value).unwrap().tensor.as_ref().unwrap().clone()),
        |index| Ok(inputs[index].clone()),
        Some(batch_size),
    )
    .unwrap()
}

pub fn run_cpu_const_operation(info: &ValueInfo, map: impl Fn(Value) -> OperationResult) -> OperationResult {
    try_run_cpu_operation(info, map, |_| Err(OperationError::MissingInput), None)
}

fn try_run_cpu_operation(
    info: &ValueInfo,
    map: impl Fn(Value) -> OperationResult,
    input: impl Fn(usize) -> OperationResult,
    batch_size: Option<usize>,
) -> OperationResult {
    let output_shape = match info.shape.as_fixed() {
        Some(shape) => shape,
        None => batch_size
            .map(|batch_size| info.shape.eval(batch_size))
            .ok_or(OperationError::NoBatchSize)?,
    };

    let output_shape_dyn = IxDyn(&output_shape.dims);

    let result: Tensor = match &info.operation {
        &Operation::Input { index } => input(index)?,
        &Operation::Constant { ref data } => {
            let data = (&**data).clone();
            Tensor::from_shape_vec(output_shape_dyn, data).unwrap()
        }
        &Operation::View { input } => {
            let input = map(input)?;
            input.reshape(output_shape_dyn)
        }
        &Operation::Broadcast { input } => {
            let input = map(input)?;
            input.broadcast(output_shape_dyn).unwrap().to_shared()
        }
        &Operation::Permute { input, ref permutation } => {
            let input = map(input)?;
            input.view().permuted_axes(permutation.clone()).to_shared()
        }
        &Operation::Slice { input, axis, range } => {
            let input = map(input)?;

            let info = slice_info(
                input.ndim(),
                axis,
                range.start as isize,
                Some(range.end as isize),
                range.step as isize,
            );
            input.slice(info).to_shared()
        }
        &Operation::Flip { input, axis } => {
            let input = map(input)?;

            // slice with negative step (ndarray convention is different from python)
            let info = slice_info(input.ndim(), axis, 0, None, -1);
            input.slice(info).to_shared()
        }
        &Operation::Gather { input, axis, indices } => {
            let input = map(input)?;
            let indices = map(indices)?;

            assert_eq!(indices.ndim(), 1);
            let slices = indices
                .iter()
                .map(|&f| {
                    let i = f as usize;
                    assert_eq!(i as f32, f);

                    input.slice(slice_info(input.ndim(), axis, i as isize, Some(i as isize + 1), 1))
                })
                .collect_vec();

            concatenate(output_shape_dyn.clone(), axis, &slices)
        }
        &Operation::Concat { ref inputs, axis } => {
            let inputs: Vec<_> = inputs.iter().map(|&x| Ok(map(x)?)).try_collect()?;
            let inputs_viewed = inputs.iter().map(|x| x.view()).collect_vec();

            concatenate(output_shape_dyn.clone(), axis, &inputs_viewed)
        }
        &Operation::Conv {
            input,
            filter,
            details: conv_shape,
        } => {
            let input = map(input)?;
            let filter = map(filter)?;
            let result = convolution(
                conv_shape,
                input.view().into_dimensionality().unwrap(),
                filter.view().into_dimensionality().unwrap(),
            );
            result.into_dyn().into_shared()
        }
        &Operation::MatMul { left, right } => {
            let left = map(left)?;
            let right = map(right)?;

            batched_mat_mul(
                left.view().into_dimensionality::<Ix3>().unwrap(),
                right.view().into_dimensionality::<Ix3>().unwrap(),
            )
            .into_dyn()
            .into_shared()
        }
        &Operation::Unary { input, op } => {
            let input = map(input)?;
            input.map(|&x| op.map(x)).into_shared()
        }
        &Operation::Binary { left, right, op } => {
            let left = map(left)?;
            let right = map(right)?;

            Zip::from(&left)
                .and(&right)
                .map_collect(|&l, &r| op.map(l, r))
                .into_shared()
        }
        &Operation::Softmax { input, axis } => {
            let input = map(input)?;
            softmax(input.view(), Axis(axis)).into_shared()
        }
        &Operation::Layernorm { input, axis, eps } => {
            let input = map(input)?;
            layernorm(input.view(), Axis(axis), eps.into_inner()).into_shared()
        }
        &Operation::Reduce { input, ref axes, op } => {
            let input = map(input)?;

            let result = axes.iter().fold(input.to_shared(), |curr, &axis| {
                Zip::from(curr.lanes(Axis(axis)))
                    .map_collect(|lane| op.reduce(lane.iter().copied()))
                    .into_shared()
                    .insert_axis(Axis(axis))
            });

            result.reshape(output_shape_dyn)
        }
    };

    assert_eq!(result.shape(), &output_shape.dims, "Wrong output shape");
    Ok(result)
}

/// Wrapper around [ndarray::concatenate] that can handle an empty input list.
fn concatenate(output_shape: IxDyn, axis: usize, inputs: &[ArrayView<f32, IxDyn>]) -> Tensor {
    if inputs.is_empty() {
        Tensor::zeros(output_shape)
    } else {
        let result = ndarray::concatenate(Axis(axis), &inputs).unwrap().into_shared();
        assert_eq!(result.dim(), output_shape);
        result
    }
}

pub fn convolution(details: ConvDetails, input: ArrayView4<f32>, kernel: ArrayView4<f32>) -> Array4<f32> {
    let ConvDetails {
        batch_size: _,
        input_channels,
        output_channels,
        input_h,
        input_w,
        kernel_h,
        kernel_w,
        stride_y,
        stride_x,
        padding_y,
        padding_x,
        output_h,
        output_w,
    } = details;

    assert!(
        kernel_h % 2 == 1 && kernel_w % 2 == 1,
        "Only odd kernels supported for now"
    );
    let batch_size = input.shape()[0];

    // We compute the convolution via im2col
    //   * create the input matrix by repeating input values around F^2 times with some padding
    //   * permute and reshape the kernel weights into a flat matrix
    //   * compute the dot product
    //   * permute and reshape the output back into a tensor
    let input_matrix = {
        let mut input_matrix = Array::zeros((batch_size, output_h, output_w, input_channels, kernel_h, kernel_w));

        // copy over entire (batch_size, input_channels) slices at once
        //   this mostly helps with non-optimized build performance, which is nice to have
        for oy in 0..output_h {
            for ox in 0..output_w {
                for fy in 0..kernel_h {
                    for fx in 0..kernel_w {
                        let iy = (oy * stride_y) as isize + fy as isize - padding_y as isize;
                        let ix = (ox * stride_x) as isize + fx as isize - padding_x as isize;

                        if (0..input_h as isize).contains(&iy) && (0..input_w as isize).contains(&ix) {
                            input_matrix
                                .slice_mut(s![.., oy, ox, .., fy, fx])
                                .assign(&input.slice(s![.., .., iy as usize, ix as isize]));
                        }
                        // leave the padding values at zero
                    }
                }
            }
        }

        input_matrix
            .into_shape((batch_size * output_h * output_w, input_channels * kernel_h * kernel_w))
            .unwrap()
    };

    let kernel_permuted = kernel.permuted_axes([1, 2, 3, 0]);
    let kernel_matrix = kernel_permuted
        .as_standard_layout()
        .into_shape((input_channels * kernel_h * kernel_w, output_channels))
        .unwrap();

    let result_matrix = input_matrix.dot(&kernel_matrix);

    let result = result_matrix
        .into_shape((batch_size, output_h, output_w, output_channels))
        .unwrap()
        .permuted_axes([0, 3, 1, 2]);

    result
}

pub fn batched_mat_mul(left: ArrayView3<f32>, right: ArrayView3<f32>) -> Array3<f32> {
    let (n0, p, q0) = left.dim();
    let (n1, q1, r) = right.dim();
    assert!(
        n0 == n1 && q0 == q1,
        "Invalid matmul dimensions: {:?} and {:?}",
        left.dim(),
        right.dim()
    );

    let mut result = Array3::zeros((n0, p, r));
    for i in 0..n0 {
        let slice = s![i, .., ..];
        result
            .slice_mut(&slice)
            .assign(&left.slice(&slice).dot(&right.slice(&slice)));
    }
    result
}

/// Softmax along the given axis of the tensor.
/// Implementation (and more importantly, the generic bounds) based on softmax within the onnxruntime crate
pub fn softmax<S, D>(array: ArrayBase<S, D>, axis: Axis) -> Array<f32, D>
where
    D: ndarray::RemoveAxis,
    S: ndarray::RawData + ndarray::Data + ndarray::RawData<Elem = f32>,
{
    let mut result = array.to_owned();

    let max = result.fold_axis(axis, f32::NEG_INFINITY, |&a, &x| a.max(x));
    result -= &max.insert_axis(axis);

    result.map_inplace(|x: &mut f32| *x = x.exp());
    let sum = result.sum_axis(axis).insert_axis(axis);
    result /= &sum;

    result
}

/// Layernorm along the given axis of the tensor.
pub fn layernorm<S, D>(array: ArrayBase<S, D>, axis: Axis, eps: f32) -> Array<f32, D>
where
    D: ndarray::RemoveAxis,
    S: ndarray::RawData + ndarray::Data + ndarray::RawData<Elem = f32>,
{
    let mut result = array.to_owned();

    let mean = result.mean_axis(axis).unwrap();
    result -= &mean.insert_axis(axis);

    let std = result
        .mapv(|f| f.powi(2))
        .mean_axis(axis)
        .unwrap()
        .mapv(|f| (f + eps).sqrt());
    result /= &std.insert_axis(axis);

    result
}

pub fn slice_info(
    rank: usize,
    axis: usize,
    start: isize,
    end: Option<isize>,
    step: isize,
) -> SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn> {
    assert_ne!(step, 0);

    let vec = (0..rank)
        .map(|r| {
            if r == axis {
                // grab the relevant range
                SliceInfoElem::Slice { start, end, step }
            } else {
                // grab everything
                SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                }
            }
        })
        .collect_vec();

    // safety: we pass an owned Vec, whose .as_ref will always return the same reference
    unsafe { SliceInfo::new(vec).unwrap() }
}

#[derive(Debug)]
pub struct ExecutionInfo {
    pub batch_size: usize,
    pub values: IndexMap<Value, CalculatedValue>,
    pub outputs: Vec<Value>,
}

pub struct CalculatedValue {
    pub value: Value,
    pub tensor: Option<Tensor>,
    pub tensor_shape: ConcreteShape,
    pub uses_seen: usize,
    pub time_spent: f32,
}

impl ExecutionInfo {
    pub fn output_tensors(self) -> Vec<Tensor> {
        self.outputs
            .iter()
            .map(|v| {
                // convert to standard layout so users get easily get &[f32] slices
                self.values
                    .get(v)
                    .unwrap()
                    .tensor
                    .as_ref()
                    .unwrap()
                    .as_standard_layout()
                    .to_shared()
            })
            .collect_vec()
    }
}

impl Debug for CalculatedValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalculatedTensor")
            .field("value", &self.value)
            .field("kept", &self.tensor.is_some())
            .field("shape", &self.tensor_shape)
            .field("time_spent", &self.time_spent)
            .finish()
    }
}

impl Display for ExecutionInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ExecutionInfo {{")?;
        for (_, value) in &self.values {
            writeln!(f, "  {:?}", value)?;
        }
        writeln!(f, "}}")?;

        Ok(())
    }
}
