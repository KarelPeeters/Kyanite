use std::collections::HashMap;

use itertools::{Itertools, zip_eq};
use ndarray::{ArcArray, Array4, ArrayView4, IxDyn, SliceInfo, SliceInfoElem};

use crate::graph::{ConvShape, Graph, Operation, Value, ValueInfo};

/// We're using an ArcArray so reshaping is free.
pub type Tensor = ArcArray<f32, IxDyn>;

pub fn execute_graph(graph: &Graph, batch_size: usize, inputs: &[&Tensor]) -> Vec<Tensor> {
    let mut map: HashMap<Value, Tensor> = HashMap::default();

    assert_eq!(graph.inputs().len(), inputs.len(), "Wrong input count");
    for (value, array) in zip_eq(graph.inputs(), inputs) {
        let shape = graph[*value].shape.eval(batch_size);
        assert_eq!(IxDyn(&shape.dims), array.dim(), "Wrong input shape");
        map.insert(*value, array.to_shared());
    }

    for output in graph.values() {
        let ValueInfo { shape, operation } = &graph[output];
        let output_shape = shape.eval(batch_size);
        let output_shape_dyn = IxDyn(&output_shape.dims);

        let result: Tensor = match operation {
            Operation::Input => continue,
            Operation::Constant { data } => {
                let data = (&**data).clone();
                Tensor::from_shape_vec(output_shape_dyn, data).unwrap()
            }
            &Operation::View { input } => {
                let input = map.get(&input).unwrap();
                input.reshape(output_shape_dyn)
            }
            &Operation::Slice { input, axis, start, end, } => {
                let input = map.get(&input).unwrap();
                let info = slice_info(input.ndim(), axis, start, end);
                input.slice(info).to_shared()
            }
            &Operation::Conv { input, filter, conv_shape } => {
                let input = map.get(&input).unwrap().view().into_dimensionality().unwrap();
                let filter = map.get(&filter).unwrap().view().into_dimensionality().unwrap();
                let result = convolution(conv_shape, input, filter);
                result.into_dyn().into_shared()
            }
            &Operation::Add { left, right } => {
                let left = map.get(&left).unwrap();
                let right = map.get(&right).unwrap();
                (left + right).into_shared()
            }
            &Operation::Mul { left, right } => {
                let left = map.get(&left).unwrap();
                let right = map.get(&right).unwrap();
                (left * right).into_shared()
            }
            &Operation::Clamp { input, min, max } => {
                let input = map.get(&input).unwrap();
                input.map(|&x| x.clamp(min, max)).into_shared()
            }
        };

        let prev = map.insert(output, result);
        assert!(prev.is_none());
    }

    graph.outputs().iter()
        .map(|output| map.get(output).unwrap().to_shared())
        .collect_vec()
}

fn convolution(shape: ConvShape, input: ArrayView4<f32>, filter: ArrayView4<f32>) -> Array4<f32> {
    let kernel_offset = shape.kernel_size / 2;
    let input_range = 0..shape.input_size;

    let output_shape = (input.dim().0, shape.output_channels, shape.output_size, shape.output_size);
    Array4::from_shape_fn(output_shape, |(n, co, ox, oy)| {
        let mut result: f32 = 0.0;

        for ci in 0..shape.input_channels {
            for kx in 0..shape.kernel_size {
                for ky in 0..shape.kernel_size {
                    let ix = ox + kx - kernel_offset;
                    let iy = oy + ky - kernel_offset;

                    result += if input_range.contains(&ix) && input_range.contains(&iy) {
                        let a = input[(n as usize, ci as usize, ix as usize, iy as usize)];
                        let f = filter[(co as usize, ci as usize, kx as usize, ky as usize)];
                        a * f
                    } else {
                        0.0
                    };
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