use std::fmt::{Debug, Formatter};
use std::time::Instant;

use convolutions_rs::convolutions::*;
use convolutions_rs::Padding;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{ArcArray, Array4, ArrayView4, concatenate, IxDyn, SliceInfo, SliceInfoElem};

use crate::graph::{ConvDetails, Graph, Operation, Value, ValueInfo};
use crate::ndarray::{Array, ArrayBase, Axis};

/// We're using an ArcArray so reshaping is free.
pub type Tensor = ArcArray<f32, IxDyn>;

pub fn cpu_execute_graph(graph: &Graph, batch_size: usize, inputs: &[&Tensor]) -> ExecutionInfo {
    assert_eq!(graph.inputs().len(), inputs.len(), "Wrong input count");

    let mut map: IndexMap<Value, CalculatedValue> = IndexMap::default();

    for output in graph.values() {
        let ValueInfo { shape, operation } = &graph[output];
        let output_shape = shape.eval(batch_size);
        let output_shape_dyn = IxDyn(&output_shape.dims);

        let start_time = Instant::now();

        let result: Tensor = match operation {
            &Operation::Input { index } => {
                inputs[index].to_shared()
            }
            Operation::Constant { data } => {
                let data = (&**data).clone();
                Tensor::from_shape_vec(output_shape_dyn, data).unwrap()
            }
            &Operation::View { input } => {
                let input = &map.get(&input).unwrap().tensor;
                input.reshape(output_shape_dyn)
            }
            &Operation::Permute { input, ref permutation } => {
                let input = &map.get(&input).unwrap().tensor;
                input.view().permuted_axes(permutation.clone()).to_shared()
            }
            &Operation::Slice { input, axis, start, end, } => {
                let input = &map.get(&input).unwrap().tensor;
                let info = slice_info(input.ndim(), axis, start, end);
                input.slice(info).to_shared()
            }
            &Operation::Gather { input, axis, indices } => {
                let input = &map.get(&input).unwrap().tensor;
                let indices = &map.get(&indices).unwrap().tensor;

                assert_eq!(indices.ndim(), 1);
                let slices = indices.iter().map(|&f| {
                    let i = f as usize;
                    assert_eq!(i as f32, f);

                    input.slice(slice_info(input.ndim(), axis, i, i + 1))
                }).collect_vec();

                concatenate(Axis(axis), &slices).unwrap().into_shared()
            }
            Operation::Concat { inputs, axis } => {
                let inputs = inputs.iter()
                    .map(|x| map.get(x).unwrap().tensor.view())
                    .collect_vec();
                ndarray::concatenate(Axis(*axis), &inputs).unwrap().into_shared()
            }
            &Operation::Conv { input, filter, details: conv_shape } => {
                let input = map.get(&input).unwrap().tensor.view().into_dimensionality().unwrap();
                let filter = map.get(&filter).unwrap().tensor.view().into_dimensionality().unwrap();
                let result = convolution(conv_shape, input, filter);
                result.into_dyn().into_shared()
            }
            &Operation::Add { left, right, subtract } => {
                let left = &map.get(&left).unwrap().tensor;
                let right = &map.get(&right).unwrap().tensor;

                let result = if subtract { left - right } else { left + right };
                result.into_shared()
            }
            &Operation::Mul { left, right } => {
                let left = &map.get(&left).unwrap().tensor;
                let right = &map.get(&right).unwrap().tensor;
                (left * right).into_shared()
            }
            &Operation::Clamp { input, min, max } => {
                let input = &map.get(&input).unwrap().tensor;
                input.map(|&x| x.clamp(min, max)).into_shared()
            }
        };

        assert_eq!(&output_shape.dims, result.shape(), "Wrong output shape");

        let end_time = Instant::now();
        let calc = CalculatedValue {
            value: output,
            tensor: result,
            time_spent: (end_time - start_time).as_secs_f32(),
        };
        let prev = map.insert(output, calc);
        assert!(prev.is_none());
    }

    ExecutionInfo {
        batch_size,
        values: map,
        outputs: graph.outputs().to_owned(),
    }
}

pub fn convolution(details: ConvDetails, input: ArrayView4<f32>, filter: ArrayView4<f32>) -> Array4<f32> {
    assert!(details.keeps_spatial_shape(), "Different in/out shape not supported yet");

    let batch_size = input.shape()[0];
    let output_shape = (batch_size, details.output_channels, details.output_h, details.output_w);

    let mut result = Array4::zeros(output_shape);
    for b in 0..batch_size {
        let result_b = conv2d(&filter, input.index_axis(Axis(0), b), Padding::Same, 1);
        result.index_axis_mut(Axis(0), b).assign(&result_b);
    }

    result
}

/// Softmax along the given axis of the tensor.
/// Implementation (and more importantly, the generic bounds) based on softmax within the onnxruntime crate
pub fn softmax<S, D>(array: ArrayBase<S, D>, axis: Axis) -> Array<f32, D>
    where
        D: ndarray::RemoveAxis,
        S: ndarray::RawData + ndarray::Data + ndarray::RawData<Elem=f32>,
{
    let mut result = array.to_owned();
    result.map_inplace(|x: &mut f32| *x = x.exp());
    let sum = result.sum_axis(axis).insert_axis(axis);
    result /= &sum;

    result
}

pub fn slice_info(rank: usize, axis: usize, start: usize, end: usize) -> SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn> {
    let vec = (0..rank)
        .map(|r| {
            if r == axis {
                // grab the relevant range
                SliceInfoElem::Slice { start: start as isize, end: Some(end as isize), step: 1 }
            } else {
                // grab everything
                SliceInfoElem::Slice { start: 0, end: None, step: 1 }
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
    pub tensor: Tensor,
    pub time_spent: f32,
}

impl ExecutionInfo {
    pub fn output_tensors(self) -> Vec<Tensor> {
        self.outputs.iter()
            .map(|v| {
                // convert to standard layout so users get easily get &[f32] slices
                self.values.get(v).unwrap().tensor
                    .as_standard_layout().to_shared()
            })
            .collect_vec()
    }
}

impl Debug for CalculatedValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CalculatedTensor")
            .field("value", &self.value)
            .field("shape", &self.tensor.dim())
            .field("time_spent", &self.time_spent)
            .finish()
    }
}
