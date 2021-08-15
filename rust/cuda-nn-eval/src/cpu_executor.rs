use std::collections::HashMap;

use itertools::{Itertools, zip_eq};
use ndarray::{Array4, ArrayView4, ArrayViewMut4};
use unwrap_match::unwrap_match;

use crate::fuser::{Activation, FusedGraph, FusedValue, FusedValueInfo};
use crate::graph::{ConvShape, Graph, Operation};

#[derive(Debug)]
pub struct CpuExecutor {
    fused_graph: FusedGraph,
    buffers: HashMap<FusedValue, Array4<f32>>,
    inputs: Vec<FusedValue>,
    outputs: Vec<FusedValue>,
}

impl CpuExecutor {
    pub fn new(graph: &Graph) -> Self {
        let fused_graph = FusedGraph::new(graph);

        let mut inputs = vec![];
        let mut buffers = HashMap::new();

        for fused_value in fused_graph.schedule() {
            let value = fused_graph[fused_value].value();
            let value_info = &graph[value];

            let [s0, s1, s2, s3] = value_info.shape;
            let shape = [s0 as usize, s1 as usize, s2 as usize, s3 as usize];
            buffers.insert(fused_value, Array4::zeros(shape));

            match &fused_graph[fused_value] {
                FusedValueInfo::Input(_) => {
                    inputs.push(fused_value)
                }
                FusedValueInfo::Constant(_) => {
                    let data = unwrap_match!(&value_info.operation, Operation::Constant {data} => data);
                    buffers.get_mut(&fused_value).unwrap().as_slice_mut().unwrap().copy_from_slice(data);
                }
                FusedValueInfo::FusedOperation { .. } => {}
            }
        }

        let outputs = graph.outputs().iter()
            .map(|&v| fused_graph.find(v))
            .collect_vec();

        CpuExecutor {
            fused_graph,
            buffers,
            inputs,
            outputs,
        }
    }

    //TODO we can easily make this dynamic in the batch sizes, we're on the CPU
    pub fn evaluate(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) {
        assert_eq!(self.inputs.len(), inputs.len(), "Wrong number of inputs");
        assert_eq!(self.outputs.len(), outputs.len(), "Wrong number of outputs");

        // copy in inputs
        for (input, value) in zip_eq(inputs, &self.inputs) {
            self.buffers.get_mut(value).unwrap().as_slice_mut().unwrap().copy_from_slice(input);
        }

        // run operations
        for output in self.fused_graph.schedule() {
            if let FusedValueInfo::FusedOperation {
                value: _,
                input, input_shape_view,
                filter, conv_shape,
                bias, res_input,
                act_mode
            } = &self.fused_graph[output] {
                let [s0, s1, s2, s3] = *input_shape_view;
                let input_shape_view = [s0 as usize, s1 as usize, s2 as usize, s3 as usize];

                //TODO assert safety
                run_convolution(
                    *conv_shape,
                    *act_mode,
                    self.buffers.get(input).unwrap().view().into_shape(input_shape_view).unwrap(),
                    self.buffers.get(filter).unwrap().view(),
                    self.buffers.get(bias).unwrap().view(),
                    res_input.as_ref().map(|res_input| self.buffers.get(res_input).unwrap().view()),
                    unsafe { (&mut *(self.buffers.get(&output).unwrap() as *const _ as *mut Array4<f32>)).view_mut() },
                )
            }
        }

        // copy outputs back
        for (output, value) in zip_eq(outputs, &self.outputs) {
            output.copy_from_slice(self.buffers.get(value).unwrap().as_slice().unwrap());
        }
    }

    pub fn fused_graph(&self) -> &FusedGraph {
        &self.fused_graph
    }

    /// Get the map containing the intermediate buffers, enables looking at internal activation after anb input was evaluated.
    pub fn buffers(&self) -> &HashMap<FusedValue, Array4<f32>> {
        &self.buffers
    }
}

fn run_convolution(
    shape: ConvShape, act_mode: Activation,
    input: ArrayView4<f32>, filter: ArrayView4<f32>, bias: ArrayView4<f32>, res: Option<ArrayView4<f32>>,
    mut output: ArrayViewMut4<f32>,
) {
    // println!("Conv {:?} x {:?} + {:?} + {:?} -> {:?}", input.shape(), filter.shape(), bias.shape(), res.map(|res| res.shape().to_vec()), output.shape());
    // println!("{:?}", shape);
    // println!("{:?}", input);
    // println!("{}", input[(0, 0, 1, 1)]);

    let kernel_offset = shape.kernel_size / 2;
    let input_range = 0..shape.input_size;

    //TODO try different loop orderings
    for n in 0..shape.batch_size {
        for ox in 0..shape.output_size {
            for oy in 0..shape.output_size {
                for k in 0..shape.output_channels {
                    let mut result: f32 = 0.0;

                    for c in 0..shape.input_channels {
                        for kx in 0..shape.kernel_size {
                            for ky in 0..shape.kernel_size {
                                let f = filter[(k as usize, c as usize, kx as usize, ky as usize)];

                                let ix = ox + kx - kernel_offset;
                                let iy = oy + ky - kernel_offset;

                                result += if input_range.contains(&ix) && input_range.contains(&iy) {
                                    let a = input[(n as usize, c as usize, ix as usize, iy as usize)];

                                    // println!("  n={} k={} c={} o=({} {}) k=({} {}) => i=({} {}) => {} * {}", n, k, c, ox, oy, kx, ky, ix, iy, a, f);

                                    a * f
                                } else {
                                    // println!("  n={} k={} c={} o=({} {}) k=({} {}) => i=({} {}) => P * {}", n, k, c, ox, oy, kx, ky, ix, iy, f);
                                    0.0
                                };
                            }
                        }
                    }

                    // println!("n={} k={} o=({} {}) => result={}", n, k, ox, oy, result);

                    result += bias[(0, k as usize, 0, 0)];
                    result += res.map_or(0.0, |res| res[(n as usize, k as usize, ox as usize, oy as usize)]);

                    let result = match act_mode {
                        Activation::Linear => result,
                        Activation::Relu => result.max(0.0),
                    };

                    output[(n as usize, k as usize, ox as usize, oy as usize)] = result;
                }
            }
        }
    }
}
