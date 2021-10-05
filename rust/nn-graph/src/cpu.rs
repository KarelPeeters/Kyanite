use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::time::Instant;

use itertools::{Itertools, zip_eq};
use ndarray::{ArcArray, Array4, ArrayView4, IxDyn, SliceInfo, SliceInfoElem};

use crate::graph::{ConvShape, Graph, Operation, Value, ValueInfo};

/// We're using an ArcArray so reshaping is free.
pub type Tensor = ArcArray<f32, IxDyn>;

pub fn cpu_execute_graph(graph: &Graph, batch_size: usize, inputs: &[&Tensor]) -> ExecutionInfo {
    let mut map: HashMap<Value, CalculatedValue> = HashMap::default();

    assert_eq!(graph.inputs().len(), inputs.len(), "Wrong input count");
    for (&value, &tensor) in zip_eq(graph.inputs(), inputs) {
        let shape = graph[value].shape.eval(batch_size);
        assert_eq!(IxDyn(&shape.dims), tensor.dim(), "Wrong input shape");

        let calc = CalculatedValue {
            value,
            tensor: tensor.to_shared(),
            time_spent: 0.0,
        };
        map.insert(value, calc);
    }

    for output in graph.values() {
        let ValueInfo { shape, operation } = &graph[output];
        let output_shape = shape.eval(batch_size);
        let output_shape_dyn = IxDyn(&output_shape.dims);

        println!("Calculating value {:?}", output);
        let start_time = Instant::now();

        let result: Tensor = match operation {
            Operation::Input => continue,
            Operation::Constant { data } => {
                let data = (&**data).clone();
                Tensor::from_shape_vec(output_shape_dyn, data).unwrap()
            }
            &Operation::View { input } => {
                let input = &map.get(&input).unwrap().tensor;
                input.reshape(output_shape_dyn)
            }
            &Operation::Slice { input, axis, start, end, } => {
                let input = &map.get(&input).unwrap().tensor;
                let info = slice_info(input.ndim(), axis, start, end);
                input.slice(info).to_shared()
            }
            &Operation::Conv { input, filter, conv_shape } => {
                let input = map.get(&input).unwrap().tensor.view().into_dimensionality().unwrap();
                let filter = map.get(&filter).unwrap().tensor.view().into_dimensionality().unwrap();
                let result = convolution(conv_shape, input, filter);
                result.into_dyn().into_shared()
            }
            &Operation::Add { left, right } => {
                let left = &map.get(&left).unwrap().tensor;
                let right = &map.get(&right).unwrap().tensor;
                (left + right).into_shared()
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
        map,
        outputs: graph.outputs().to_owned(),
    }
}

fn convolution(shape: ConvShape, input: ArrayView4<f32>, filter: ArrayView4<f32>) -> Array4<f32> {
    let kernel_offset = (shape.kernel_size / 2) as isize;
    let input_range = 0..shape.input_size as isize;

    let output_shape = (input.dim().0, shape.output_channels, shape.output_size, shape.output_size);
    Array4::from_shape_fn(output_shape, |(n, co, ox, oy)| {
        let mut result: f32 = 0.0;

        for ci in 0..shape.input_channels {
            for kx in 0..shape.kernel_size as isize {
                for ky in 0..shape.kernel_size as isize {
                    let ix = ox as isize + kx - kernel_offset;
                    let iy = oy as isize + ky - kernel_offset;

                    if input_range.contains(&ix) && input_range.contains(&iy) {
                        let a = input[(n, ci, ix as usize, iy as usize)];
                        let f = filter[(co, ci, kx as usize, ky as usize)];

                        result += a * f
                    }
                }
            }
        }

        result
    })
}

fn slice_info(rank: usize, axis: usize, start: usize, end: usize) -> SliceInfo<Vec<SliceInfoElem>, IxDyn, IxDyn> {
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
    map: HashMap<Value, CalculatedValue>,
    outputs: Vec<Value>,
}

pub struct CalculatedValue {
    value: Value,
    tensor: Tensor,
    time_spent: f32,
}

impl ExecutionInfo {
    pub fn outputs(self) -> Vec<Tensor> {
        self.outputs.iter()
            .map(|v| self.map.get(v).unwrap().tensor.to_shared())
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