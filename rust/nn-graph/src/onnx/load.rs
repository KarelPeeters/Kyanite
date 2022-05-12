use byteorder::{ByteOrder, LittleEndian};
use itertools::{zip_eq, Itertools};
use num_traits::cast;
use prost::Message;

pub use crate::graph::Graph;
use crate::graph::{BinaryOp, ReduceOp, SliceRange, UnaryOp};
use crate::onnx::attributes::Attributes;
use crate::onnx::proto::tensor_proto::DataType;
use crate::onnx::proto::tensor_shape_proto::dimension::Value as ProtoDimValue;
use crate::onnx::proto::type_proto::Value as ProtoTypeValue;
use crate::onnx::proto::{tensor_shape_proto, ModelProto, TensorProto, TypeProto};
use crate::onnx::store::Store;
use crate::onnx::typed_value::{SizeOrInt, TypedValue};
use crate::shape;
use crate::shape::{Shape, Size};

pub fn load_model_proto(buf: &[u8]) -> ModelProto {
    let mut buf: &[u8] = buf;
    ModelProto::decode(&mut buf).unwrap()
}

pub fn onnx_proto_to_graph(model: &ModelProto) -> Graph {
    let model_graph = model.graph.as_ref().unwrap();

    // TODO rethink how all of this works, it's getting very tedious now
    let mut graph = Graph::new();
    let mut nodes: Store<TypedValue> = Store::default();

    load_initializers(&mut graph, &mut nodes, &model_graph.initializer);

    for input in &model_graph.input {
        // initializers are allowed to re-appear in the inputs, so we skip them the second time
        if nodes.contains(&*input.name) {
            continue;
        }

        let shape = resolve_float_tensor_shape(input.r#type.as_ref().unwrap());
        let value = graph.input(shape);
        nodes.define(&input.name, TypedValue::FloatTensor(value));
    }

    for node in &model_graph.node {
        assert_eq!(1, node.output.len(), "nodes with multiple outputs not yet supported");
        let output_name = &node.output[0];

        let mut attrs = Attributes::from(&node.attribute);
        let inputs: Vec<&TypedValue> = node
            .input
            .iter()
            .enumerate()
            .map(|(i, name)| {
                nodes
                    .get(name)
                    .unwrap_or_else(|| panic!("Input {} {} of node {} not found", i, name, node.name))
            })
            .collect_vec();

        // TODO put this huge match expression in a separate function
        // TODO in general panic a lot less and return a proper result type instead
        let op_type = &*node.op_type;
        let value: TypedValue = match op_type {
            "Conv" => {
                assert!(inputs.len() <= 3);
                let input = inputs[0].unwrap_float();
                let filter = inputs[1].unwrap_float();
                let bias_raw = inputs.get(2).map(|v| v.unwrap_float());

                let groups = attrs.take_int("group");
                let kernel_shape = attrs.take_ints("kernel_shape");
                let padding = attrs.take_ints("pads");
                let strides = attrs.take_ints("strides");
                let dilations = attrs.take_ints("dilations");

                let filter_shape = graph[filter]
                    .shape
                    .unwrap_fixed("Convolution kernel shape must be fixed");

                // always add bias in the 2D conv view domain, so it's easier to fuse later on
                let bias = bias_raw.map(|bias| {
                    let bias_size = graph[bias].shape.unwrap_1();
                    let bias_view_shape = shape![1, bias_size, 1, 1];

                    graph.view(bias, bias_view_shape)
                });

                assert_eq!(1, groups);
                let conv_rank = kernel_shape.len();

                let result = match conv_rank {
                    1 => {
                        let kernel_size0 = unwrap_1(kernel_shape);
                        let [padding_0, padding_1] = unwrap_2(padding);
                        let stride = unwrap_1(strides);
                        let dilation = unwrap_1(dilations);

                        let [_, _, kernel_size1] = filter_shape.unwrap_3();

                        assert_eq!(padding_0, padding_1);
                        assert!(dilation == 1 && stride == 1);
                        assert_eq!(kernel_size0, kernel_size1);

                        let input_extra = graph.view(input, graph[input].shape.clone().concat(&shape![1]));
                        let filter_extra = graph.view(filter, graph[filter].shape.clone().concat(&shape![1]));

                        let result_conv = graph.conv(input_extra, filter_extra, padding_0, 0);
                        let result_biased = bias.map_or(result_conv, |bias| graph.add(result_conv, bias));

                        let result_shape = graph[result_biased].shape.replace(3, None);
                        let result = graph.view(result_biased, result_shape);

                        result
                    }
                    2 => {
                        let [kernel_h0, kernel_w0] = unwrap_2(kernel_shape);
                        let [padding_y0, padding_x0, padding_y1, padding_x1] = unwrap_4(padding);
                        let [stride_y, stride_x] = unwrap_2(strides);
                        let [dilation_y, dilation_x] = unwrap_2(dilations);

                        let [_, _, kernel_h1, kernel_w1] = filter_shape.unwrap_4();

                        assert!(padding_y0 == padding_y1 && padding_x0 == padding_x1 && padding_y0 == padding_x0);
                        assert!(dilation_y == 1 && dilation_x == 1);
                        assert!(stride_y == 1 && stride_x == 1);
                        assert!(kernel_h1 == kernel_h0 && kernel_w1 == kernel_w0);

                        let result_conv = graph.conv(input, filter, padding_y0, padding_x0);
                        let result_biased = bias.map_or(result_conv, |bias| graph.add(result_conv, bias));

                        result_biased
                    }
                    rank => panic!("{}d convolution not yet supported", rank),
                };

                TypedValue::FloatTensor(result)
            }
            "Relu" => {
                assert_eq!(1, inputs.len());
                let input = inputs[0].unwrap_float();
                let result = graph.relu(input);
                TypedValue::FloatTensor(result)
            }
            "Clip" => {
                assert!(!inputs.is_empty());
                let input = inputs[0].unwrap_float();

                let (min, max) = match inputs.len() {
                    1 => (attrs.take_float("min"), attrs.take_float("max")),
                    3 => {
                        let min = graph.as_const(inputs[1].unwrap_float()).unwrap();
                        let max = graph.as_const(inputs[1].unwrap_float()).unwrap();

                        assert!(
                            min.len() == 1 && max.len() == 1,
                            "Expected min and max to be a single element, got {} and {}",
                            min.len(),
                            max.len(),
                        );

                        (min[0], max[0])
                    }
                    len => panic!("Expected either 1 or 3 inputs for Clip, got {}", len),
                };

                let result = graph.clamp(input, min, max);
                TypedValue::FloatTensor(result)
            }
            "Sqrt" => {
                let op = match op_type {
                    "Sqrt" => UnaryOp::Sqrt,
                    _ => unreachable!(),
                };

                assert_eq!(1, inputs.len());
                let input = inputs[0].unwrap_float();

                let result = graph.unary(op, input);
                TypedValue::FloatTensor(result)
            }
            "Add" | "Sub" | "Mul" | "Div" | "Min" | "Max" | "Pow" => {
                let op = match op_type {
                    "Add" => BinaryOp::Add,
                    "Sub" => BinaryOp::Sub,
                    "Mul" => BinaryOp::Mul,
                    "Div" => BinaryOp::Div,
                    "Min" => BinaryOp::Min,
                    "Max" => BinaryOp::Max,
                    "Pow" => BinaryOp::Pow,
                    _ => unreachable!(),
                };

                assert_eq!(2, inputs.len());
                let left = inputs[0];
                let right = inputs[1];

                if let (Some(left), Some(right)) = (left.as_shape(&graph), right.as_shape(&graph)) {
                    // if they're both shapes keep it that way
                    assert_eq!(left.len(), right.len());

                    let run_op = match op {
                        BinaryOp::Mul => |a: Size, b: Size| a * b,
                        BinaryOp::Div => {
                            |a: Size, b: Size| (a / b).unwrap_or_else(|| panic!("Failed to divide {:?} by {:?}", a, b))
                        }
                        _ => panic!("Unsupported shape operation {:?}", op),
                    };

                    let result = zip_eq(left, right)
                        .map(|(l, r)| SizeOrInt::Size(run_op(l.as_size().unwrap(), r.as_size().unwrap())))
                        .collect_vec();
                    TypedValue::Shape(result)
                } else if let (TypedValue::FloatTensor(left), TypedValue::FloatTensor(right)) = (left, right) {
                    // if they're both float tensors just add a graph operation
                    let result = graph.binary(op, *left, *right);
                    TypedValue::FloatTensor(result)
                } else if let (TypedValue::Shape(left), TypedValue::FloatTensor(right)) = (left, right) {
                    assert_eq!(left.len(), 1);
                    assert_eq!(graph[*right].shape, Shape::SCALAR);

                    let left_value = left[0].as_size().unwrap().unwrap_fixed("ele left hand size") as f32;
                    let right_value = *graph.as_const(*right).unwrap().iter().next().unwrap();

                    let result_value = op.map(left_value, right_value);

                    let result = graph.constant(Shape::SCALAR, vec![result_value]);
                    TypedValue::FloatTensor(result)
                } else {
                    panic!(
                        "Elementwise operation between {:?} and {:?} not implemented for node {:?}",
                        left, right, output_name,
                    )
                }
            }
            "Flatten" => {
                assert_eq!(1, inputs.len());
                let input = inputs[0].unwrap_float();

                let rel_axis = attrs.take_int("axis");
                let axis = abs_axis(rel_axis, graph[input].shape.rank());

                let result = graph.flatten(input, axis);
                TypedValue::FloatTensor(result)
            }
            "Gemm" => {
                assert_eq!(3, inputs.len());

                let input = inputs[0].unwrap_float();
                let weight = inputs[1].unwrap_float();
                let bias = inputs.get(2).map(|v| v.unwrap_float());

                let alpha = attrs.take_float("alpha");
                let beta = attrs.take_float("beta");
                let trans_b = attrs.take_int("transB") != 0;

                assert_eq!(1.0, alpha);
                assert_eq!(1.0, beta);
                assert!(trans_b);

                let linear = graph.linear(input, weight);

                let result = if let Some(bias) = bias {
                    let bias_len = graph[bias].shape.unwrap_1();
                    let bias_view_shape = shape![1, bias_len];
                    let bias_view = graph.view(bias, bias_view_shape);

                    graph.add(linear, bias_view)
                } else {
                    linear
                };

                TypedValue::FloatTensor(result)
            }
            "MatMul" => {
                assert_eq!(2, inputs.len());

                let input_a = inputs[0].unwrap_float();
                let input_b = inputs[1].unwrap_float();

                let result = graph.mat_mul(input_a, input_b);
                TypedValue::FloatTensor(result)
            }
            "BatchNormalization" => {
                //TODO also try without merging anything here to see how much of a difference it makes

                assert_eq!(5, inputs.len());

                let input = inputs[0].unwrap_float();

                // assume everything is constant for now, so we can immediately fuse stuff
                let scale = graph.as_const(inputs[1].unwrap_float()).unwrap();
                let bias = graph.as_const(inputs[2].unwrap_float()).unwrap();
                let mean = graph.as_const(inputs[3].unwrap_float()).unwrap();
                let variance = graph.as_const(inputs[4].unwrap_float()).unwrap();

                let epsilon = attrs.take_float("epsilon");
                let _ = attrs.take_float("momentum");

                // figure out the shapes
                let input_shape = &graph[input].shape;
                assert!(input_shape.rank() >= 2, "BN input must have at least rank 2");
                let index = 1;
                let const_shape = input_shape.keep(index, Size::ONE);

                let channels = input_shape[1].unwrap_fixed("BN channel count must be fixed");
                assert!(
                    scale.len() == channels
                        && bias.len() == channels
                        && mean.len() == channels
                        && variance.len() == channels
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
                let result = graph.add(scaled, total_bias);
                TypedValue::FloatTensor(result)
            }
            "Constant" => {
                assert!(inputs.is_empty());

                let tensor = attrs.take_tensor("value");
                define_tensor_data(&mut graph, tensor)
            }
            "Cast" => {
                assert_eq!(1, inputs.len());

                let input = inputs[0];
                let to_type = DataType::from_i32(attrs.take_int("to") as i32).expect("Invalid data type");

                match input {
                    // ignore casting for shapes
                    TypedValue::Shape(shape) => TypedValue::Shape(shape.clone()),
                    _ => {
                        let input = input.unwrap_tensor();
                        match to_type {
                            DataType::Float => TypedValue::FloatTensor(input),
                            DataType::Int64 => TypedValue::IntTensor(input),
                            _ => panic!("Casting to type {:?} not supported yet", to_type),
                        }
                    }
                }
            }
            "Reshape" => {
                assert_eq!(2, inputs.len());

                let input = inputs[0];
                let new_shape = inputs[1].as_shape(&graph).unwrap();

                let input_tensor = input.unwrap_tensor();
                let input_size = graph[input_tensor].shape.size();
                let output_shape = calculate_reshape_output_shape(input_size, &new_shape);

                let result = graph.view(input_tensor, output_shape);
                TypedValue::with_same_type(result, input)
            }
            "Unsqueeze" => {
                assert_eq!(1, inputs.len());

                let input = inputs[0];
                let rel_axes = attrs.take_ints("axes");

                match input {
                    // shapes are shapeless, so just return the same value
                    TypedValue::Shape(shape) => TypedValue::Shape(shape.clone()),
                    // actual unsqueeze
                    _ => {
                        let input_tensor = input.unwrap_tensor();
                        let input_shape = &graph[input_tensor].shape;

                        let output_rank = input_shape.rank() + rel_axes.len();
                        let axes = rel_axes.iter().map(|&a| abs_axis(a, output_rank)).collect_vec();

                        assert!(
                            axes.iter().all_unique() && axes.iter().all(|&a| a < output_rank),
                            "Invalid axis {:?} for input rank {} in Unsqueeze",
                            axes,
                            input_shape.rank(),
                        );

                        let mut input_shape_left = input_shape.dims.iter().copied();
                        let output_dims = (0..output_rank)
                            .map(|i| {
                                if axes.contains(&i) {
                                    Size::ONE
                                } else {
                                    input_shape_left.next().unwrap()
                                }
                            })
                            .collect_vec();
                        assert_eq!(input_shape_left.len(), 0);

                        let output_shape = Shape::new(output_dims);
                        let result = graph.view(input_tensor, output_shape);

                        TypedValue::with_same_type(result, input)
                    }
                }
            }
            "Transpose" => {
                assert_eq!(1, inputs.len());
                let input = inputs[0];

                let permutation = attrs.take_ints("perm");
                let permutation = permutation.iter().map(|&x| x as usize).collect_vec();

                let result = graph.permute(input.unwrap_tensor(), permutation);
                TypedValue::with_same_type(result, input)
            }
            "Gather" => {
                assert_eq!(2, inputs.len());

                let input = inputs[0];
                let indices = inputs[1].unwrap_int();

                let rel_axis = attrs.maybe_take_int("axis").unwrap_or(0);
                assert!(rel_axis >= 0, "Negative gather axis not supported yet");
                let axis = rel_axis as usize;

                match graph[indices].shape.rank() {
                    0 => {
                        if let Some(index) = graph.as_const(indices) {
                            assert_eq!(index.len(), 1);
                            let index = *index.iter().next().unwrap() as usize;

                            match input {
                                TypedValue::Shape(dims) => TypedValue::Shape(vec![dims[index]]),
                                TypedValue::FloatTensor(input_tensor) | TypedValue::IntTensor(input_tensor) => {
                                    TypedValue::with_same_type(graph.index(*input_tensor, axis, index), input)
                                }
                            }
                        } else {
                            panic!("Rank-0 gather only supported with constant index");
                        }
                    }
                    1 => TypedValue::with_same_type(graph.gather(input.unwrap_tensor(), axis, indices), input),
                    rank => {
                        panic!("Gather only supported for index rank <2, got {}", rank)
                    }
                }
            }
            "Slice" => {
                let (input, starts, ends, axes, steps) = match inputs.len() {
                    1 => {
                        let input = inputs[0];

                        let starts = attrs.take_ints("starts").to_vec();
                        let ends = attrs.take_ints("ends").to_vec();
                        let axes = attrs.maybe_take_ints("axes").map(|v| v.to_vec());
                        let steps = attrs.maybe_take_ints("steps").map(|v| v.to_vec());

                        (input, starts, ends, axes, steps)
                    }
                    3 | 4 | 5 => {
                        let unwrap = |v: &&TypedValue| {
                            graph
                                .as_const(v.unwrap_int())
                                .unwrap()
                                .iter()
                                .map(|&f| f as i64)
                                .collect_vec()
                        };

                        let input = inputs[0];
                        let starts = unwrap(&inputs[1]);
                        let ends = unwrap(&inputs[2]);
                        let axes = inputs.get(3).map(unwrap);
                        let steps = inputs.get(4).map(unwrap);

                        (input, starts, ends, axes, steps)
                    }
                    len => panic!("Unexpected number of arguments to Slice operator: {}", len),
                };

                let slice_rank = starts.len();
                let axes = axes.unwrap_or_else(|| (0..slice_rank as i64).collect_vec());
                let steps = steps.unwrap_or_else(|| vec![1; slice_rank]);

                assert!(
                    slice_rank == ends.len() && slice_rank == axes.len() && slice_rank == steps.len(),
                    "Inconsistent axes count"
                );
                assert!(
                    axes.iter().all_unique(),
                    "Slice axis cannot be repeated, got {:?}",
                    axes
                );

                match input {
                    TypedValue::Shape(shape) => {
                        assert_eq!(slice_rank, 1, "Shape slicing can only happen on a single axis");
                        assert_eq!(axes[0], 0, "Shape slicing can only happen along axis 0");
                        assert_eq!(steps[0], 1, "Shape slicing only works with step 1 for now");

                        let start = abs_axis(starts[0], shape.len());
                        let end = abs_axis(ends[0], shape.len());

                        TypedValue::Shape(shape[start..end].to_vec())
                    }
                    &TypedValue::FloatTensor(input_tensor) | &TypedValue::IntTensor(input_tensor) => {
                        let input_shape = graph[input_tensor].shape.clone();

                        let result = (0..slice_rank).fold(input_tensor, |curr, i| {
                            let axis = abs_axis(axes[i], input_shape.rank());
                            let axis_size = input_shape[axis].unwrap_fixed("Slice axis size");

                            let step = steps[i];
                            assert_ne!(step, 0, "Step cannot be 0");

                            if step > 0 {
                                let start = abs_axis(starts[i], axis_size);
                                let end = abs_axis(ends[i], axis_size);

                                let range = SliceRange::new(start, end, step as usize);
                                graph.slice(curr, axis, range)
                            } else {
                                assert!(
                                    starts[i] == -1 && ends[i] == i64::MIN && steps[i] == -1,
                                    "Only simple flip negative stride supported for now"
                                );
                                graph.flip(curr, axis)
                            }
                        });
                        TypedValue::with_same_type(result, input)
                    }
                }
            }
            "Concat" => {
                assert!(!inputs.is_empty(), "Must concatenate at least one value");
                let rank = inputs[0].shape(&graph).rank();

                let rel_axis = attrs.take_int("axis");
                let axis = abs_axis(rel_axis, rank);

                let any_shape = inputs.iter().any(|x| matches!(x, TypedValue::Shape(_)));
                let any_float = inputs.iter().any(|x| matches!(x, TypedValue::FloatTensor(_)));
                let any_int = inputs.iter().any(|x| matches!(x, TypedValue::IntTensor(_)));

                if any_shape {
                    assert_eq!(axis, 0, "Shape concatenation must happen along axis 0");

                    let shape = inputs
                        .iter()
                        .flat_map(|x| x.as_shape(&graph).unwrap().into_iter())
                        .collect_vec();

                    TypedValue::Shape(shape)
                } else {
                    let input_tensors = inputs.iter().map(|v| v.unwrap_tensor()).collect_vec();
                    let result = graph.concat(input_tensors, axis, None);

                    if any_float {
                        assert!(
                            inputs.iter().all(|&x| matches!(x, TypedValue::FloatTensor(_))),
                            "All concatenated values must have the same type (float)"
                        );
                        TypedValue::FloatTensor(result)
                    } else if any_int {
                        assert!(
                            inputs.iter().all(|&x| matches!(x, TypedValue::IntTensor(_))),
                            "All concatenated values must have the same type (int)"
                        );
                        TypedValue::IntTensor(result)
                    } else {
                        unreachable!()
                    }
                }
            }
            "Shape" => {
                assert_eq!(1, inputs.len());
                let input = inputs[0].unwrap_tensor();
                let shape = graph[input].shape.clone();
                let dims = shape.dims.iter().copied().map(SizeOrInt::Size).collect_vec();
                TypedValue::Shape(dims)
            }
            "Softmax" => {
                assert_eq!(1, inputs.len());
                let input = inputs[0].unwrap_float();

                let shape = graph[input].shape.clone();
                let axis = attrs.maybe_take_int("axis").unwrap_or(-1);
                let axis = abs_axis(axis, shape.rank());

                TypedValue::FloatTensor(graph.softmax(input, axis))
            }
            "ReduceSum" | "ReduceMean" | "ReduceProd" | "ReduceMin" | "ReduceMax" => {
                let op = match op_type {
                    "ReduceSum" => ReduceOp::Sum,
                    "ReduceMean" => ReduceOp::Mean,
                    "ReduceProd" => ReduceOp::Prod,
                    "ReduceMin" => ReduceOp::Min,
                    "ReduceMax" => ReduceOp::Max,
                    _ => unreachable!(),
                };

                assert_eq!(1, inputs.len());
                let input = inputs[0].unwrap_float();
                let input_shape = graph[input].shape.clone();

                let axes = attrs.maybe_take_ints("axes").map_or_else(
                    || (0..input_shape.rank()).collect_vec(),
                    |axes| axes.iter().map(|&a| abs_axis(a, input_shape.rank())).collect_vec(),
                );
                let keep_dims = attrs.maybe_take_int("keepdims").unwrap_or(1) != 0;

                let result_shape = if keep_dims {
                    input_shape.replace_all(&axes, Some(Size::ONE))
                } else {
                    input_shape.clone()
                };

                let result = graph.reduce(input, axes, op);
                let result_shaped = graph.view(result, result_shape);

                TypedValue::FloatTensor(result_shaped)
            }
            _ => {
                eprintln!("Already parsed graph:\n{:?}", graph);
                panic!("Unsupported op_type '{}' in node {}", op_type, node.name);
            }
        };

        // register operation output as node
        nodes.define(output_name, value);
        assert!(attrs.is_done(), "Leftover attributes: {:?}", attrs);
    }

    for output in &model_graph.output {
        graph.output(nodes[&output.name].unwrap_float());
    }

    graph
}

fn load_initializers<'a>(graph: &mut Graph, store: &mut Store<'a, TypedValue>, initializers: &'a [TensorProto]) {
    for tensor in initializers {
        let value = define_tensor_data(graph, tensor);
        store.define(&tensor.name, value)
    }
}

fn define_tensor_data(graph: &mut Graph, tensor: &TensorProto) -> TypedValue {
    // figure out the shape
    let dims = tensor.dims.iter().map(|&d| Size::fixed(d as usize)).collect_vec();
    let shape = Shape::new(dims);
    let size = shape.size().unwrap_fixed("Data tensor shape must be fixed");

    // load the data
    let data_type = DataType::from_i32(tensor.data_type).expect("Illegal data type");

    let (is_int, data) = match data_type {
        DataType::Float => {
            let data = if !tensor.float_data.is_empty() {
                tensor.float_data.clone()
            } else {
                let mut float_data = vec![0.0; size];
                LittleEndian::read_f32_into(&tensor.raw_data, &mut float_data);
                float_data
            };

            (false, data)
        }
        DataType::Double => {
            let data = if !tensor.double_data.is_empty() {
                tensor.double_data.iter().map(|&f| f as f32).collect_vec()
            } else {
                let mut data = vec![0.0; size];
                LittleEndian::read_f64_into(&tensor.raw_data, &mut data);
                data.iter().map(|&d| cast(d).unwrap()).collect_vec()
            };

            (false, data)
        }
        DataType::Int64 => {
            let data = if !tensor.int64_data.is_empty() {
                tensor.int64_data.iter().map(|&i| cast(i).unwrap()).collect_vec()
            } else {
                let mut data = vec![0; size];
                LittleEndian::read_i64_into(&tensor.raw_data, &mut data);
                data.iter().map(|&i| cast(i).unwrap()).collect_vec()
            };

            (true, data)
        }
        _ => panic!("Unexpected data type {:?} {}", data_type, tensor.data_type),
    };

    let value = graph.constant(shape, data);

    match is_int {
        false => TypedValue::FloatTensor(value),
        true => TypedValue::IntTensor(value),
    }
}

fn resolve_float_tensor_shape(ty: &TypeProto) -> Shape {
    let value = ty.value.as_ref().expect("Value doesn't have type set");
    match value {
        ProtoTypeValue::TensorType(tensor) => {
            assert_eq!(
                tensor.elem_type,
                DataType::Float as i32,
                "only floats supported for now"
            );

            let dims = tensor
                .shape
                .as_ref()
                .expect("Tensor does not have shape set")
                .dim
                .iter()
                .map(resolve_tensor_dim)
                .collect_vec();
            Shape::new(dims)
        }
        _ => panic!("Unsupported value kind {:?}", value),
    }
}

fn resolve_tensor_dim(dim: &tensor_shape_proto::Dimension) -> Size {
    let value = dim.value.as_ref().expect("Missing value for dimension");

    match value {
        &ProtoDimValue::DimValue(inner) => Size::fixed(inner as usize),
        ProtoDimValue::DimParam(name) => {
            assert_eq!(name, "batch_size");
            Size::BATCH
        }
    }
}

fn abs_axis(axis: i64, rank: usize) -> usize {
    if axis == i64::MAX {
        rank
    } else if axis < 0 {
        rank - ((-axis) as usize)
    } else {
        axis as usize
    }
}

#[track_caller]
fn unwrap_1(slice: &[i64]) -> usize {
    assert_eq!(slice.len(), 1, "Expected 1 element, got {:?}", slice);
    slice[0] as usize
}

#[track_caller]
fn unwrap_2(slice: &[i64]) -> [usize; 2] {
    assert_eq!(slice.len(), 2, "Expected 2 elements, got {:?}", slice);
    [slice[0] as usize, slice[1] as usize]
}

#[track_caller]
fn unwrap_4(slice: &[i64]) -> [usize; 4] {
    assert_eq!(slice.len(), 4, "Expected 4 elements, got {:?}", slice);
    [
        slice[0] as usize,
        slice[1] as usize,
        slice[2] as usize,
        slice[3] as usize,
    ]
}

fn calculate_reshape_output_shape(old_size: Size, new_shape_raw: &[SizeOrInt]) -> Shape {
    let mut new_shape = vec![];
    let mut leftover_index = None;
    let mut leftover_size = old_size;

    for (i, &size_or_int) in new_shape_raw.iter().enumerate() {
        let size = match size_or_int {
            SizeOrInt::Int(-1) => {
                assert!(
                    leftover_index.is_none(),
                    "Reshape shape can only contain a single -1 value"
                );
                leftover_index = Some(i);
                new_shape.push(Size::ZERO);
                continue;
            }
            SizeOrInt::Int(int) => {
                assert!(int >= 0, "Size must be positive or -1");
                Size::fixed(int as usize)
            }
            SizeOrInt::Size(size) => size,
        };

        leftover_size = (leftover_size / size).unwrap_or_else(|| {
            panic!("Cannot reshape {} into {:?}", old_size, new_shape_raw);
        });
        new_shape.push(size);
    }

    if let Some(leftover_index) = leftover_index {
        new_shape[leftover_index] = leftover_size;
    }

    let shape = Shape::new(new_shape);
    assert_eq!(old_size, shape.size(), "Output and input sizes differ");

    shape
}
