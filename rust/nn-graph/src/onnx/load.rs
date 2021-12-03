use byteorder::{ByteOrder, LittleEndian};
use itertools::Itertools;
use num_traits::cast;
use prost::Message;

use crate::graph::{Graph, Value};
use crate::onnx::attributes::Attributes;
use crate::onnx::proto::{ModelProto, tensor_shape_proto, TensorProto, TypeProto};
use crate::onnx::proto::tensor_proto::DataType;
use crate::onnx::proto::tensor_shape_proto::dimension::Value as ProtoDimValue;
use crate::onnx::proto::type_proto::Value as ProtoTypeValue;
use crate::onnx::store::Store;
use crate::shape;
use crate::shape::{Shape, Size};

pub fn load_model_proto(buf: &[u8]) -> ModelProto {
    let mut buf: &[u8] = &buf;
    let model = ModelProto::decode(&mut buf)
        .unwrap();
    model
}

pub fn onnx_proto_to_graph(model: &ModelProto) -> Graph {
    let model_graph = model.graph.as_ref().unwrap();

    let mut graph = Graph::new();
    let mut nodes = Store::default();

    load_initializers(&mut graph, &mut nodes, &model_graph.initializer);

    for input in &model_graph.input {
        // initializers are allowed to re-appear in the inputs, so we skip them the second time
        if nodes.contains(&&*input.name) {
            continue;
        }

        let shape = resolve_tensor_shape(input.r#type.as_ref().unwrap());
        let value = graph.input(shape);
        nodes.define(&input.name, value);
    }

    for node in &model_graph.node {
        assert_eq!(1, node.output.len(), "nodes with multiple outputs not supported");
        let output_name = &node.output[0];

        let mut attrs = Attributes::from(&node.attribute);
        let inputs: Vec<Value> = node.input.iter().enumerate().map(|(i, name)| {
            *nodes.get(name)
                .unwrap_or_else(|| panic!("Input {} {} of node {} not found", i, name, node.name))
        }).collect_vec();

        let value = match &*node.op_type {
            "Conv" => {
                assert!(inputs.len() <= 3);
                let input = inputs[0];
                let filter = inputs[1];
                let bias = inputs.get(2).copied();

                let g = attrs.take_int("group");
                let [kw, kh] = unwrap_2(attrs.take_ints("kernel_shape"));
                let [ph0, pv0, ph1, pv1] = unwrap_4(attrs.take_ints("pads"));
                let [sw, sh] = unwrap_2(attrs.take_ints("strides"));
                let [dw, dh] = unwrap_2(attrs.take_ints("dilations"));

                let [_, _, kernel_w, kernel_h] =
                    graph[filter].shape.unwrap_fixed("Convolution kernel shape must be fixed").unwrap_4();

                assert_eq!(1, g);
                assert!(ph0 == ph1 && pv0 == pv1 && ph0 == pv0);
                assert!(dw == 1 && dh == 1);
                assert!(sw == 1 && sh == 1);
                assert!(kernel_w == kw && kernel_h == kh);

                let conv = graph.conv(input, filter, ph0);

                if let Some(bias) = bias {
                    let bias_size = graph[bias].shape.unwrap_1();
                    let bias_view_shape = shape![1, bias_size, 1, 1];

                    let bias_view = graph.view(bias, bias_view_shape);
                    graph.add(conv, bias_view)
                } else {
                    conv
                }
            }
            "Relu" => {
                assert_eq!(1, inputs.len());
                let input = inputs[0];
                graph.relu(input)
            }
            "Clip" => {
                assert!(inputs.len() >= 1);
                let input = inputs[0];

                let (min, max) = match inputs.len() {
                    1 =>
                        (attrs.take_float("min"), attrs.take_float("max")),
                    3 => {
                        let min = graph.as_const(inputs[1]).unwrap();
                        let max = graph.as_const(inputs[1]).unwrap();

                        assert!(
                            min.len() == 1 && max.len() == 1,
                            "Expected min and max to be a single element, got {} and {}",
                            min.len(), max.len(),
                        );

                        (min[0], max[0])
                    }
                    len =>
                        panic!("Expected either 1 or 3 inputs for Clip, got {}", len),
                };

                graph.clamp(input, min, max)
            }
            "Add" => {
                assert_eq!(2, inputs.len());
                let left = inputs[0];
                let right = inputs[1];
                graph.add(left, right)
            }
            "Flatten" => {
                assert_eq!(1, inputs.len());
                let input = inputs[0];

                let rel_axis = attrs.take_int("axis");
                let axis = index_to_abs(rel_axis, graph[input].shape.rank());

                graph.flatten(input, axis)
            }
            "Gemm" => {
                assert_eq!(3, inputs.len());

                let input = inputs[0];
                let weight = inputs[1];
                let bias = inputs.get(2).copied();

                let alpha = attrs.take_float("alpha");
                let beta = attrs.take_float("beta");
                let trans_b = attrs.take_int("transB") != 0;

                assert_eq!(1.0, alpha);
                assert_eq!(1.0, beta);
                assert!(trans_b);

                let linear = graph.linear(input, weight);

                let output = if let Some(bias) = bias {
                    let bias_len = graph[bias].shape.unwrap_1();
                    let bias_view_shape = shape![1, bias_len];
                    let bias_view = graph.view(bias, bias_view_shape);

                    graph.add(linear, bias_view)
                } else {
                    linear
                };

                output
            }
            "BatchNormalization" => {
                //TODO also try without merging anything here to see how much of a difference it makes

                assert_eq!(5, inputs.len());

                let input = inputs[0];

                // assume everything is constant for now, so we can immediately fuse stuff
                let scale = graph.as_const(inputs[1]).unwrap();
                let bias = graph.as_const(inputs[2]).unwrap();
                let mean = graph.as_const(inputs[3]).unwrap();
                let variance = graph.as_const(inputs[4]).unwrap();

                let epsilon = attrs.take_float("epsilon");
                let _ = attrs.take_float("momentum");

                // figure out the shapes
                let input_shape = &graph[input].shape;
                assert!(input_shape.rank() >= 2, "BN input must have at least rank 2");
                let const_shape = input_shape.all_ones_except(1);

                let channels = input_shape[1].unwrap_fixed("BN channel count must be fixed");
                assert!(
                    scale.len() == channels && bias.len() == channels &&
                        mean.len() == channels && variance.len() == channels
                );

                // fuse everything into a single scale and bias
                let total_scale = (0..channels)
                    .map(|i| scale[i] / (variance[i] + epsilon).sqrt())
                    .collect_vec();
                let total_bias = (0..channels)
                    .map(|i| bias[i] - (mean[i] * scale[i] / (variance[i] + epsilon).sqrt()))
                    .collect_vec();

                // put everything into the graph
                let total_scale = graph.constant(const_shape.clone(), total_scale);
                let total_bias = graph.constant(const_shape.clone(), total_bias);

                let scaled = graph.mul(input, total_scale);
                graph.add(scaled, total_bias)
            }
            "Constant" => {
                assert!(inputs.is_empty());

                let tensor = attrs.take_tensor("value");
                let (shape, data) = load_tensor_float_data(tensor);

                graph.constant(shape, data)
            }
            "Cast" => {
                assert_eq!(1, inputs.len());
                let input = inputs[0];

                let to_type = DataType::from_i32(attrs.take_int("to") as i32)
                    .expect("Invalid data type");
                assert_eq!(to_type, DataType::Int64);

                graph.as_const(input).unwrap();

                // we don't actually cast anything here, and casting is just up to the user
                //  just make sure we're casting a const to int so nothing can go terribly wrong
                input
            }
            "Reshape" => {
                assert_eq!(2, inputs.len());

                let input = inputs[0];
                let new_shape = inputs[1];

                assert_eq!(1, graph[new_shape].shape.rank(), "Reshape shape must have rank 1");
                let new_shape_f = graph.as_const(new_shape).unwrap();

                let input_shape = &graph[input].shape;
                let output_shape = calculate_reshape_output_shape(input_shape.size(), new_shape_f);

                graph.view(input, output_shape)
            }
            "Gather" => {
                assert_eq!(2, inputs.len());

                let input = inputs[0];
                let indices = inputs[1];

                let axis = attrs.take_int("axis") as usize;

                assert_eq!(graph[indices].shape.rank(), 0, "Only single index gather supported for now");
                let index = graph.as_const(indices).unwrap()[0] as usize;

                graph.index(input, axis, index)
            }
            "Slice" => {
                assert_eq!(1, inputs.len());
                let input = inputs[0];
                let input_shape = graph[input].shape.clone();

                let axes = attrs.take_ints("axes");
                let starts = attrs.take_ints("starts");
                let ends = attrs.take_ints("ends");

                assert!(axes.len() == starts.len() && axes.len() == ends.len(), "Inconsistent axes count");

                (0..axes.len()).fold(input, |curr, i| {
                    let axis = index_to_abs(axes[i], input_shape.rank());
                    let axis_size = input_shape[axis].unwrap_fixed("Slice axis size");

                    graph.slice(
                        curr,
                        axis,
                        index_to_abs(starts[i], axis_size),
                        index_to_abs(ends[i], axis_size),
                    )
                })
            }
            _ => {
                eprintln!("Already parsed graph:\n{:?}", graph);
                panic!("Unsupported op_type '{}' in node {}", node.op_type, node.name);
            }
        };

        // register operation output as node
        nodes.define(&output_name, value);
        assert!(attrs.is_done(), "Leftover attributes: {:?}", attrs);
    }

    for output in &model_graph.output {
        graph.output(nodes[&output.name]);
    }

    graph
}

fn load_initializers<'a>(graph: &mut Graph, store: &mut Store<'a, Value>, initializers: &'a [TensorProto]) {
    for tensor in initializers {
        let (shape, data) = load_tensor_float_data(tensor);
        let node = graph.constant(shape, data);
        store.define(&tensor.name, node)
    }
}

fn load_tensor_float_data(tensor: &TensorProto) -> (Shape, Vec<f32>) {
    // figure out the dimension
    let dims = tensor.dims.iter().map(|&d| Size::fixed(d as usize)).collect_vec();
    let shape = Shape::new(dims);
    let size = shape.size().unwrap_fixed("Data tensor shape must be fixed");

    // load the data
    let data_type = DataType::from_i32(tensor.data_type).expect("Illegal data type");

    let data = match data_type {
        DataType::Float => {
            if !tensor.float_data.is_empty() {
                tensor.float_data.clone()
            } else {
                let mut float_data = vec![0.0; size];
                LittleEndian::read_f32_into(&tensor.raw_data, &mut float_data);
                float_data
            }
        }
        DataType::Double => {
            if !tensor.double_data.is_empty() {
                tensor.double_data.iter().map(|&f| f as f32).collect_vec()
            } else {
                let mut data = vec![0.0; size];
                LittleEndian::read_f64_into(&tensor.raw_data, &mut data);
                data.iter().map(|&d| cast(d).unwrap()).collect_vec()
            }
        }
        //TODO it's really starting to become easier to just implement types in the graph
        DataType::Int64 => {
            if !tensor.int64_data.is_empty() {
                tensor.int64_data.iter().map(|&i| cast(i).unwrap()).collect_vec()
            } else {
                let mut data = vec![0; size];
                LittleEndian::read_i64_into(&tensor.raw_data, &mut data);
                data.iter().map(|&i| cast(i).unwrap()).collect_vec()
            }
        }
        _ => panic!("Unexpected data type {:?} {}", data_type, tensor.data_type),
    };

    (shape, data)
}

fn resolve_tensor_shape(ty: &TypeProto) -> Shape {
    let value = ty.value.as_ref().expect("Value doesn't have type set");
    match value {
        ProtoTypeValue::TensorType(tensor) => {
            assert_eq!(tensor.elem_type, DataType::Float as i32, "only floats supported for now");

            let dims = tensor.shape.as_ref()
                .expect("Tensor does not have shape set")
                .dim.iter()
                .map(|d| resolve_tensor_dim(d))
                .collect_vec();
            Shape::new(dims)
        }
        _ => panic!("Unsupported value kind {:?}", value),
    }
}

fn resolve_tensor_dim(dim: &tensor_shape_proto::Dimension) -> Size {
    let value = dim.value.as_ref()
        .expect("Missing value for dimension");

    match value {
        &ProtoDimValue::DimValue(inner) => Size::fixed(inner as usize),
        ProtoDimValue::DimParam(name) => {
            assert_eq!(name, "batch_size");
            Size::BATCH
        }
    }
}

fn index_to_abs(index: i64, size: usize) -> usize {
    if index == i64::MAX {
        size
    } else if index < 0 {
        size - ((-index) as usize)
    } else {
        index as usize
    }
}

#[track_caller]
fn unwrap_2(slice: &[i64]) -> [usize; 2] {
    assert_eq!(slice.len(), 2, "Expected 2 elements, got {:?}", slice);
    [slice[0] as usize, slice[1] as usize]
}

#[track_caller]
fn unwrap_4(slice: &[i64]) -> [usize; 4] {
    assert_eq!(slice.len(), 4, "Expected 4 elements, got {:?}", slice);
    [slice[0] as usize, slice[1] as usize, slice[2] as usize, slice[3] as usize]
}

fn calculate_reshape_output_shape(old_size: Size, new_shape_f: &[f32]) -> Shape {
    let new_shape_int = new_shape_f.iter().map(|&f| {
        assert_eq!(f as i64 as f32, f, "Reshape shape must only contain integers, got {}", f);
        f as i64
    }).collect_vec();

    let mut new_shape = vec![];
    let mut leftover_index = None;
    let mut leftover_size = old_size;

    for (i, &size) in new_shape_int.iter().enumerate() {
        let size = if size == -1 {
            assert!(leftover_index.is_none(), "Reshape shape can only contain a single -1 value");
            leftover_index = Some(i);
            Size::ZERO
        } else {
            assert!(size >= 0, "Size must be positive or -1");
            let size = Size::fixed(size as usize);
            leftover_size = (leftover_size / size).unwrap_or_else(|| {
                panic!("Cannot reshape {} into {:?}", old_size, new_shape_int);
            });
            size
        };

        new_shape.push(size);
    }

    if let Some(leftover_index) = leftover_index {
        new_shape[leftover_index] = leftover_size;
    }

    let shape = Shape::new(new_shape);
    assert_eq!(old_size, shape.size(), "Output and input sizes differ");

    shape
}
