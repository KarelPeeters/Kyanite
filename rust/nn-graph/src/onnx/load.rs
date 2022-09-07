use byteorder::{ByteOrder, LittleEndian};
use itertools::{zip_eq, Itertools};
use num_traits::cast;
use prost::Message;

pub use crate::graph::Graph;
use crate::graph::{BinaryOp, ReduceOp, SliceRange, UnaryOp};
use crate::onnx::inputs::{Attributes, Inputs};
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

    graph.take_new_values();

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

        let mut attrs = Attributes::from(node.name.clone(), &node.attribute);
        let mut inputs = Inputs::from(node.name.clone(), &node.input, &nodes);

        // TODO put this huge match expression in a separate function
        // TODO in general panic a lot less and return a proper result type instead
        let op_type = &*node.op_type;
        let value: TypedValue = match op_type {
            "Conv" => {
                let input = inputs.required(0).unwrap_float();
                let filter = inputs.required(1).unwrap_float();
                let bias_raw = inputs.optional(2).map(|v| v.unwrap_float());

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

                        let result_conv = graph.conv(input_extra, filter_extra, 1, 1, padding_0, 0);
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
                        assert!(kernel_h1 == kernel_h0 && kernel_w1 == kernel_w0);

                        let result_conv = graph.conv(input, filter, stride_y, stride_x, padding_y0, padding_x0);
                        let result_biased = bias.map_or(result_conv, |bias| graph.add(result_conv, bias));

                        result_biased
                    }
                    rank => panic!("{}d convolution not yet supported", rank),
                };

                TypedValue::FloatTensor(result)
            }
            "Relu" => {
                let input = inputs.required(0).unwrap_float();
                let result = graph.relu(input);
                TypedValue::FloatTensor(result)
            }
            "Clip" => {
                let input = inputs.required(0).unwrap_float();
                // these are optional since the older version of the operator used attributes instead
                let input_min = inputs.optional(1);
                let input_max = inputs.optional(2);

                let (min, max) = match (input_min, input_max) {
                    (None, None) => (attrs.take_float("min"), attrs.take_float("max")),
                    (Some(min), Some(max)) => {
                        let min = graph.as_const(min.unwrap_float()).unwrap();
                        let max = graph.as_const(max.unwrap_float()).unwrap();

                        assert!(
                            min.len() == 1 && max.len() == 1,
                            "Expected min and max to be a single element, got {} and {}",
                            min.len(),
                            max.len(),
                        );

                        (min[0], max[0])
                    }
                    _ => panic!("Clip must have either 1 or 3 inputs, got 2",),
                };

                let result = graph.clamp(input, min, max);
                TypedValue::FloatTensor(result)
            }
            "Sqrt" | "Exp" | "Sigmoid" => {
                let op = match op_type {
                    "Sqrt" => UnaryOp::Sqrt,
                    "Exp" => UnaryOp::Exp,
                    "Sigmoid" => UnaryOp::Sigmoid,
                    _ => unreachable!(),
                };

                let input = inputs.required(0).unwrap_float();
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

                let left = inputs.required(0);
                let right = inputs.required(1);

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
                let input = inputs.required(0).unwrap_float();

                let rel_axis = attrs.take_int("axis");
                let axis = abs_axis(rel_axis, graph[input].shape.rank());

                let result = graph.flatten(input, axis);
                TypedValue::FloatTensor(result)
            }
            "Gemm" => {
                let input = inputs.required(0).unwrap_float();
                let weight = inputs.required(1).unwrap_float();
                let bias = inputs.optional(2).map(|v| v.unwrap_float());

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
                let left = inputs.required(0).unwrap_float();
                let right = inputs.required(1).unwrap_float();

                // TODO we're still missing support for 1D operand broadcasting, but that should be pretty rare
                let result = graph.mat_mul(left, right);
                TypedValue::FloatTensor(result)
            }
            "BatchNormalization" => {
                //TODO also try without merging anything here to see how much of a difference it makes

                let input = inputs.required(0).unwrap_float();

                // assume everything is constant for now, so we can immediately fuse stuff
                let scale = graph.as_const(inputs.required(1).unwrap_float()).unwrap();
                let bias = graph.as_const(inputs.required(2).unwrap_float()).unwrap();
                let mean = graph.as_const(inputs.required(3).unwrap_float()).unwrap();
                let variance = graph.as_const(inputs.required(4).unwrap_float()).unwrap();

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
            "InstanceNormalization" => {
                let input = inputs.required(0).unwrap_float();
                let scale = inputs.required(1).unwrap_float();
                let bias = inputs.required(2).unwrap_float();
                let epsilon = attrs.take_float("epsilon");

                let shape = graph[input].shape.clone();
                assert!(
                    shape.rank() >= 2,
                    "Input rank must be >= 2, for the the batch and channel axes, got {}",
                    shape
                );

                let rest_size = shape.dims[2..].iter().copied().product::<Size>();
                let flat_shape = shape![shape[0], shape[1], rest_size];
                let broadcast_shape = Shape::ones(shape.rank()).replace(1, Some(shape[1]));

                let flat = graph.view(input, flat_shape);
                let norm_flat = graph.layernorm(flat, 2, epsilon);
                let norm = graph.view(norm_flat, shape);

                let scale_broadcast = graph.view(scale, broadcast_shape.clone());
                let bias_broadcast = graph.view(bias, broadcast_shape);

                let scaled = graph.mul(norm, scale_broadcast);
                let result = graph.add(scaled, bias_broadcast);

                TypedValue::FloatTensor(result)
            }
            "Constant" => {
                let tensor = attrs.take_tensor("value");
                define_tensor_data(&mut graph, tensor)
            }
            "ConstantOfShape" => {
                let shape = inputs.optional(0);

                let shape = match shape {
                    None => Shape::SCALAR,
                    Some(shape) => Shape::new(
                        shape
                            .as_shape(&graph)
                            .unwrap()
                            .iter()
                            .map(|s| s.as_size().unwrap())
                            .collect_vec(),
                    ),
                };

                let value = match attrs.maybe_take_tensor("value") {
                    None => TypedValue::FloatTensor(graph.constant(Shape::SCALAR, vec![0.0])),
                    Some(tensor) => define_tensor_data(&mut graph, tensor),
                };

                assert_eq!(
                    value.shape(&graph).size(),
                    Size::ONE,
                    "value must be a one-element tensor"
                );

                let scalar = graph.view(value.unwrap_tensor(), Shape::SCALAR);
                let result = graph.broadcast(scalar, shape);
                TypedValue::with_same_type(result, &value)
            }
            "Cast" => {
                let input = inputs.required(0);
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
                let input = inputs.required(0);
                let new_shape = inputs.required(1).as_shape(&graph).unwrap();
                let allow_zero = attrs.maybe_take_bool("allowzero").unwrap_or(false);

                let input_tensor = input.unwrap_tensor();
                let old_shape = &graph[input_tensor].shape;
                let output_shape = calculate_reshape_output_shape(old_shape, &new_shape, allow_zero);

                let result = graph.view(input_tensor, output_shape);
                TypedValue::with_same_type(result, input)
            }
            "Unsqueeze" => {
                let input = inputs.required(0);

                let rel_axes = match inputs.optional(1) {
                    Some(rel_axes) => {
                        let axes = rel_axes.unwrap_const_int(&graph);
                        assert_eq!(
                            axes.shape().len(),
                            1,
                            "Unsqueeze axes must be 1D tensor, got shape {:?}",
                            axes.shape(),
                        );
                        axes.iter().copied().collect_vec()
                    }
                    None => attrs.take_ints("axes").to_vec(),
                };

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
                let input = inputs.required(0);

                let permutation = attrs.take_ints("perm");
                let permutation = permutation.iter().map(|&x| x as usize).collect_vec();

                let result = graph.permute(input.unwrap_tensor(), permutation);
                TypedValue::with_same_type(result, input)
            }
            "Gather" => {
                let input = inputs.required(0);
                let indices = inputs.required(1).unwrap_int();

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
                let get = |inputs: &mut Inputs, attrs: &mut Attributes, index: usize, name: &str| match inputs
                    .optional(index)
                {
                    Some(value) => {
                        let value = value.unwrap_const_int(&graph);
                        assert_eq!(
                            value.shape().len(),
                            1,
                            "Slice operand {} must be 1D const, got shape {:?}",
                            name,
                            value.shape()
                        );
                        Some(value.iter().copied().collect_vec())
                    }
                    None => attrs.maybe_take_ints(name).map(|v| v.to_vec()),
                };

                let input = inputs.required(0);
                let starts = get(&mut inputs, &mut attrs, 1, "starts").expect("Missing starts input and attribute");
                let ends = get(&mut inputs, &mut attrs, 2, "ends").expect("Missing ends input and attribute");

                let slice_rank = starts.len();
                let axes =
                    get(&mut inputs, &mut attrs, 3, "axes").unwrap_or_else(|| (0..slice_rank as i64).collect_vec());
                let steps = get(&mut inputs, &mut attrs, 4, "steps").unwrap_or_else(|| vec![1; slice_rank]);

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
                let inputs = inputs.take_all_variadic();
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
            "Pad" => {
                // operands
                let input = inputs.required(0).unwrap_float();
                let pads = inputs.required(1).unwrap_const_int(&graph);
                let constant_value = inputs.optional(2);
                let axes = inputs.optional(3);
                let mode = attrs.maybe_take_string("mode").unwrap_or("constant");

                // map operands
                let input_shape = &graph[input].shape;

                let constant_value = constant_value
                    .map(|v| graph.as_single_const(v.unwrap_float()).unwrap())
                    .unwrap_or(0.0);

                let axes = match axes {
                    Some(axes) => {
                        let axes = axes.unwrap_const_int(&graph);
                        assert_eq!(
                            axes.shape().len(),
                            1,
                            "Axes must be 1D tensor, got shape {:?}",
                            axes.shape()
                        );
                        axes.iter().map(|&i| abs_axis(i, input_shape.rank())).collect_vec()
                    }
                    None => (0..input_shape.rank()).collect_vec(),
                };

                assert_eq!(pads.shape(), &[axes.len() * 2], "Pads and axes shape mismatch");
                let pads = pads.iter().copied().collect_vec();

                assert_eq!(mode, "constant", "Only 'constant' pad mode supported");

                let constant = graph.constant(Shape::SCALAR, vec![constant_value]);

                // TODO consider adding a real pad operation instead of this concat workaround
                let output = axes.iter().fold(input, |acc, &axis| {
                    let acc_shape = graph[acc].shape.clone();

                    let pad_left = pads[axis];
                    let pad_right = pads[axes.len() + axis];
                    assert!(pad_left >= 0 && pad_right >= 0, "Pad values cannot be negative");

                    let blocks = vec![
                        graph.broadcast(constant, acc_shape.replace(axis, Some(Size::fixed(pad_left as usize)))),
                        acc,
                        graph.broadcast(constant, acc_shape.replace(axis, Some(Size::fixed(pad_right as usize)))),
                    ];
                    graph.concat(blocks, axis, None)
                });

                TypedValue::FloatTensor(output)
            }
            "Shape" => {
                let input = inputs.required(0).unwrap_tensor();
                let shape = graph[input].shape.clone();
                let dims = shape.dims.iter().copied().map(SizeOrInt::Size).collect_vec();
                TypedValue::Shape(dims)
            }
            "Identity" => {
                let input = inputs.required(0);
                input.clone()
            }
            "Softmax" => {
                let input = inputs.required(0).unwrap_float();

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

                let input = inputs.required(0).unwrap_float();
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
            "Resize" => {
                // operands
                let input = inputs.required(0);
                let roi = inputs.optional(1);
                let scales = inputs.optional(2).map(|v| v.unwrap_float());
                let sizes = inputs.optional(3).map(|v| v.unwrap_int());

                let _antialias = attrs.maybe_take_bool("antialias").unwrap_or(false);
                let axes = attrs.maybe_take_ints("axes");
                let _coordinate_transformation_mode = attrs
                    .maybe_take_string("coordinate_transformation_mode")
                    .unwrap_or("half_pixel");
                let _cubic_coeff_a = attrs.take_float("cubic_coeff_a");
                let _exclude_outside = attrs.maybe_take_int("exclude_outside").unwrap_or(0);
                let _extrapolation_value = attrs.maybe_take_float("extrapolation_value").unwrap_or(0.0);
                let keep_aspect_ratio_policy = attrs
                    .maybe_take_string("keep_aspect_ratio_policy")
                    .unwrap_or("stretch")
                    .to_owned();
                let mode = attrs.maybe_take_string("mode").unwrap_or("nearest").to_owned();
                let nearest_mode = attrs
                    .maybe_take_string("nearest_mode")
                    .unwrap_or("round_prefer_floor")
                    .to_owned();

                // require exactly matching operands for most
                assert!(
                    mode == "nearest"
                        && nearest_mode == "floor"
                        && roi.is_none()
                        && sizes.is_none()
                        && axes.is_none()
                        && keep_aspect_ratio_policy == "stretch",
                    "The given resize operation is not supported"
                );

                let scales = graph
                    .as_const(scales.expect("Resize requires scales for now"))
                    .expect("Resize only supported with constant scales");

                let input_tensor = input.unwrap_tensor();
                let input_shape = &graph[input_tensor].shape;
                let rank = input_shape.rank();

                assert_eq!(
                    scales.shape(),
                    &[rank],
                    "Scales must be a vector with length the input rank"
                );

                let result = scales.iter().enumerate().fold(input_tensor, |acc, (i, &scale_f)| {
                    let scale = scale_f as usize;
                    assert_eq!(scale as f32, scale_f, "Only integer scales supported, got {:?}", scales);
                    graph.repeat(acc, i, scale)
                });

                TypedValue::with_same_type(result, input)
            }
            _ => {
                eprintln!("Already parsed graph:\n{}", graph);
                panic!("Unsupported op_type '{}' in node {}", op_type, node.name);
            }
        };

        // set debug id for all newly created nodes to the current node name
        for value in graph.take_new_values() {
            graph.set_debug_id(value, node.name.clone())
        }

        // check that we used all attributes and inputs
        assert!(
            attrs.is_done(),
            "Node {} has leftover attributes: {:?}",
            node.name,
            attrs
        );
        let leftover_inputs = inputs.leftover_inputs();
        if leftover_inputs.len() > 0 {
            panic!("Node {} has leftover inputs: {:?}", node.name, leftover_inputs);
        }

        // actually define the current node
        nodes.define(output_name, value);
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

fn calculate_reshape_output_shape(old_shape: &Shape, new_shape_raw: &[SizeOrInt], allow_zero: bool) -> Shape {
    let old_size = old_shape.size();

    let mut new_shape = vec![];
    let mut leftover_index = None;
    let mut leftover_size = old_size;

    for (i, &size_or_int) in new_shape_raw.iter().enumerate() {
        let size = match size_or_int {
            SizeOrInt::Int(0) => {
                if allow_zero {
                    Size::ZERO
                } else {
                    assert!(
                        i < old_shape.rank(),
                        "Cannot copy dim {} of output shape {:?}, not present in input {}",
                        i,
                        new_shape,
                        old_shape,
                    );
                    old_shape[i]
                }
            }
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
