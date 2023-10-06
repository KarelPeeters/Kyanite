use std::path::PathBuf;

use byteorder::{ByteOrder, LittleEndian};
use itertools::Itertools;
use ndarray::{Axis, azip};
use prost::Message;

use crate::shape;
use crate::cpu::{cpu_flip, cpu_gather, cpu_slice};
use crate::dtype::{DTensor, DType, Tensor};
use crate::graph::{BinaryOp, broadcast_shape_symmetric, broadcast_tensors_symmetric, ReduceOp, SliceRange, UnaryOp, Value};
pub use crate::graph::Graph;
use crate::onnx::external_data::ExternalDataLoader;
use crate::onnx::inputs::{Attributes, Inputs};
use crate::onnx::proto::{ModelProto, tensor_shape_proto, TensorProto, TypeProto};
use crate::onnx::proto::tensor_proto::DataLocation;
use crate::onnx::proto::tensor_proto::DataType;
use crate::onnx::proto::tensor_shape_proto::dimension::Value as ProtoDimValue;
use crate::onnx::proto::type_proto::Value as ProtoTypeValue;
use crate::onnx::result::{Node, OnnxError, OnnxResult, UnwrapProto};
use crate::onnx::store::Store;
use crate::onnx::typed_value::{OnnxValue, SignedSize};
use crate::shape::{Shape, Size};

// TODO convert every possible panic to an error (even in the shape classes if possible)
// we use &dyn to avoid duplicate codegen of this large and non-critical function
pub fn graph_from_onnx_bytes(buf: &[u8], external: &mut dyn ExternalDataLoader) -> OnnxResult<Graph> {
    let model = load_model_proto(buf);
    let model_graph = model.graph.as_ref().unwrap_proto("model.graph")?;

    let mut graph = Graph::new();
    let mut nodes: Store<OnnxValue> = Store::default();

    // load initializer values (similar to constants but defined separately)
    for tensor in &model_graph.initializer {
        let value = define_tensor_data(&mut graph, &tensor.name, tensor, external)?;
        nodes.define(&tensor.name, OnnxValue::Value(value))
    }

    // clear newly defined values so we don't attribute them to the first node
    graph.take_new_values();

    for input in &model_graph.input {
        // initializers are allowed to re-appear in the inputs, so we skip them the second time
        if nodes.contains(&input.name) {
            continue;
        }

        let (shape, dtype) = resolve_tensor_type(input.r#type.as_ref().unwrap_proto("input.type")?)?;
        let value = graph.input(shape, dtype);
        nodes.define(&input.name, OnnxValue::Value(value));
    }

    for node_proto in &model_graph.node {
        let node = Node {
            name: node_proto.name.as_str(),
            op_type: node_proto.op_type.as_str(),
        };

        if node_proto.output.len() != 1 {
            return Err(OnnxError::UnsupportedMultipleOutputs(
                node.to_owned(),
                node_proto.output.clone(),
            ));
        }
        let output_name = &node_proto.output[0];

        let mut attrs = Attributes::from(node, &node_proto.attribute);
        let mut inputs = Inputs::from(node, &node_proto.input, &nodes)?;

        let value: OnnxValue = visit_node(&mut graph, external, node, &mut inputs, &mut attrs)?;

        // set debug id for all newly created nodes to the current node name
        for value in graph.take_new_values() {
            graph.set_debug_id(value, node.name.to_owned())
        }

        // check that the value if only a size if necessary
        value.assert_valid();

        // check that we used all attributes and inputs
        let leftover_attributes = attrs.leftover();
        if !leftover_attributes.is_empty() {
            return Err(OnnxError::LeftoverAttributes(node.to_owned(), leftover_attributes));
        }
        let leftover_inputs = inputs.leftover();
        if !leftover_inputs.is_empty() {
            return Err(OnnxError::LeftoverInputs(node.to_owned(), leftover_inputs));
        }

        // actually define the current node
        nodes.define(output_name, value);
    }

    for output in &model_graph.output {
        let value_or_size = &nodes[output.name.as_str()];
        let value = value_or_size.unwrap_value().ok_or(OnnxError::ExpectedNonBatchValue(output.name.clone()))?;
        graph.output(value);
    }

    Ok(graph)
}

fn visit_node(
    graph: &mut Graph,
    external: &mut dyn ExternalDataLoader,
    node: Node<&str>,
    inputs: &mut Inputs,
    attrs: &mut Attributes,
) -> OnnxResult<OnnxValue> {
    let result = match node.op_type {
        "Conv" => {
            let input = inputs.required(0)?.unwrap_value().unwrap();
            let filter = inputs.required(1)?.unwrap_value().unwrap();
            let bias_raw = inputs.optional(2).map(|v| v.unwrap_value().unwrap());

            let groups = attrs.take_int("group")?;
            let kernel_shape = attrs.take_ints("kernel_shape")?;
            let padding = attrs.take_ints("pads")?;
            let strides = attrs.take_ints("strides")?;
            let dilations = attrs.take_ints("dilations")?;

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

                    let result_shape = graph[result_biased].shape.replace(3, shape![]);
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
                rank => return Err(OnnxError::UnsupportedNdConvolution(node.to_owned(), rank)),
            };

            OnnxValue::Value(result)
        }
        "Clip" => {
            let input = inputs.required(0)?.unwrap_value().unwrap();
            // these are optional since the older version of the operator used attributes instead
            let input_min = inputs.optional(1);
            let input_max = inputs.optional(2);

            let result = match (input_min, input_max) {
                (None, None) => {
                    let min = attrs.take_float("min")?;
                    let max = attrs.take_float("max")?;
                    graph.clamp::<f32>(input, min, max)
                }
                (Some(min), Some(max)) => {
                    let min = min.unwrap_value().unwrap();
                    let max = max.unwrap_value().unwrap();
                    assert_eq!(graph[min].shape, Shape::SCALAR);
                    assert_eq!(graph[max].shape, Shape::SCALAR);

                    let mid = graph.binary(BinaryOp::Min, input, max);
                    let result = graph.binary(BinaryOp::Max, mid, min);
                    result
                }
                _ => {
                    let message = "Clip must have either 1 or 3 inputs, got 2".to_owned();
                    return Err(OnnxError::InvalidOperationArgs(node.to_owned(), message));
                }
            };

            OnnxValue::Value(result)
        }
        "Abs" | "Neg" | "Sin" | "Cos" | "Exp" | "Log" | "Sqrt" | "Sigmoid" | "Relu" | "Tanh" | "Erf" => {
            let input = inputs.required(0)?.unwrap_value().unwrap();

            let result = match node.op_type {
                "Abs" => graph.unary(UnaryOp::Abs, input),
                "Neg" => graph.unary(UnaryOp::Neg, input),
                "Sin" => graph.unary(UnaryOp::Sin, input),
                "Cos" => graph.unary(UnaryOp::Cos, input),
                "Exp" => graph.unary(UnaryOp::Exp, input),
                "Log" => graph.unary(UnaryOp::Log, input),
                "Sqrt" => graph.unary(UnaryOp::Sqrt, input),
                "Sigmoid" => graph.unary(UnaryOp::Sigmoid, input),
                "Relu" => graph.relu(input),
                "Tanh" => graph.unary(UnaryOp::Tanh, input),
                "Erf" => graph.unary(UnaryOp::Erf, input),
                _ => unreachable!(),
            };

            OnnxValue::Value(result)
        }
        "Add" | "Sub" | "Mul" | "Div" | "Min" | "Max" | "Pow" => {
            let op = match node.op_type {
                "Add" => BinaryOp::Add,
                "Sub" => BinaryOp::Sub,
                "Mul" => BinaryOp::Mul,
                "Div" => BinaryOp::Div,
                "Min" => BinaryOp::Min,
                "Max" => BinaryOp::Max,
                "Pow" => BinaryOp::Pow,
                _ => unreachable!(),
            };

            let left = inputs.required(0)?;
            let right = inputs.required(1)?;

            if let (&OnnxValue::Value(left), &OnnxValue::Value(right)) = (left, right) {
                // keep values as values
                OnnxValue::Value(graph.binary(op, left, right))
            } else {
                // decay to shape
                let left = left.as_size(graph)?;
                let right = right.as_size(graph)?;

                let (left, right) = broadcast_tensors_symmetric(&left, &right);

                let result = azip!(&left, &right).map_collect(|&l, &r| {
                    eval_binary_op(op, l, r).unwrap_or_else(|| panic!("Operation {:?} failed between {:?} and {:?}", op, left, right))
                });

                // the batch size might have cancelled out!
                OnnxValue::new_size(result.into_shared(), graph)
            }
        }
        "Equal" => {
            todo!("implement eq again, once we have proper bool types (and have decided to stop worrying about batch_size in onnx)")
        }
        "Where" => {
            todo!("implement once we have bool types")
        }
        "Flatten" => {
            let input = inputs.required(0)?;

            // figure out the axis
            let rel_axis = attrs.maybe_take_int("axis")?.unwrap_or(1);
            let axis = abs_axis(rel_axis, input.shape(graph).rank());

            // figure out new shape
            let kept_shape = &input.shape(graph).dims[..axis];
            let flat_shape = input.shape(graph).dims[axis..].iter().copied().product::<Size>();

            let mut new_shape = kept_shape.to_vec();
            new_shape.push(flat_shape);

            // strange special case in onnx spec, insert additional 1 axis
            if axis == 0 {
                new_shape.insert(0, Size::ONE);
            }
            let new_shape = Shape::new(new_shape);

            // apply view operation
            match input {
                &OnnxValue::Value(input) =>
                    OnnxValue::Value(graph.view(input, new_shape)),
                OnnxValue::Size(input) =>
                    OnnxValue::new_size(input.reshape(new_shape.unwrap_fixed("size shape").dims), graph),
            }
        }
        "Gemm" => {
            let input = inputs.required(0)?.unwrap_value().unwrap();
            let weight = inputs.required(1)?.unwrap_value().unwrap();
            let bias = inputs.optional(2).map(|v| v.unwrap_value().unwrap());

            let alpha = attrs.take_float("alpha")?;
            let beta = attrs.take_float("beta")?;
            let trans_b = attrs.take_int("transB")? != 0;

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

            OnnxValue::Value(result)
        }
        "MatMul" => {
            let left = inputs.required(0)?.unwrap_value().unwrap();
            let right = inputs.required(1)?.unwrap_value().unwrap();

            // TODO we're still missing support for 1D operand broadcasting, but that should be pretty rare
            let result = graph.mat_mul(left, right);
            OnnxValue::Value(result)
        }
        "Einsum" => {
            let inputs = inputs.take_all_variadic();
            let equation = attrs.take_string("equation")?;

            let equation_compact = equation.replace(' ', "");

            // TODO for now we hardcode some typical einsum operations, replace this with a general implementation
            //   look into "tensor primitives" and optimal "contractions"?
            match equation_compact.as_ref() {
                "bid,bjd->bij" => {
                    assert_eq!(inputs.len(), 2);
                    let left = inputs[0].unwrap_value().unwrap();
                    let right = inputs[1].unwrap_value().unwrap();

                    let right_transpose = graph.permute(right, vec![0, 2, 1]);
                    let result = graph.batched_mat_mul(left, right_transpose);
                    OnnxValue::Value(result)
                }
                "bij,bjd->bid" => {
                    assert_eq!(inputs.len(), 2);
                    let left = inputs[0].unwrap_value().unwrap();
                    let right = inputs[1].unwrap_value().unwrap();

                    let result = graph.batched_mat_mul(left, right);
                    OnnxValue::Value(result)
                }
                _ => panic!(
                    "Einsum with inputs equation {:?} and inputs {:?} not yet supported",
                    equation, inputs
                ),
            }
        }
        // TODO ensure the optimizer can fuse the scale/eps/var and mean/bias operations
        "BatchNormalization" => {
            let input = inputs.required(0)?.unwrap_value().unwrap();

            let input_scale = inputs.required(1)?.unwrap_value().unwrap();
            let input_bias = inputs.required(2)?.unwrap_value().unwrap();
            let input_mean = inputs.required(3)?.unwrap_value().unwrap();
            let input_variance = inputs.required(4)?.unwrap_value().unwrap();

            let epsilon = attrs.take_float("epsilon")?;
            let _ = attrs.take_float("momentum")?;

            // figure out the shapes
            let input_shape = &graph[input].shape;
            assert!(input_shape.rank() >= 2, "BN input must have at least rank 2");

            let channels = input_shape[1];
            let shape_vec = shape![channels];
            let shape_exp = input_shape.keep(1, Size::ONE);

            for param in [input_scale, input_bias, input_mean, input_variance] {
                assert_eq!(graph[param].shape, shape_vec);
            }

            // put everything into the graph
            let result = {
                let value_eps = graph.scalar(epsilon);

                let exp_scale = graph.view(input_scale, shape_exp.clone());
                let exp_bias = graph.view(input_bias, shape_exp.clone());
                let exp_mean = graph.view(input_mean, shape_exp.clone());
                let exp_variance = graph.view(input_variance, shape_exp);

                let div_squared = graph.add(exp_variance, value_eps);
                let div = graph.unary(UnaryOp::Sqrt, div_squared);

                let x = input;
                let x_mean = graph.sub(x, exp_mean);
                let x_div = graph.binary(BinaryOp::Div, x_mean, div);
                let x_scale = graph.mul(x_div, exp_scale);
                let x_bias = graph.add(x_scale, exp_bias);
                x_bias
            };

            OnnxValue::Value(result)
        }
        "InstanceNormalization" => {
            let input = inputs.required(0)?.unwrap_value().unwrap();
            let scale = inputs.required(1)?.unwrap_value().unwrap();
            let bias = inputs.required(2)?.unwrap_value().unwrap();
            let epsilon = attrs.take_float("epsilon")?;

            let shape = graph[input].shape.clone();
            assert!(
                shape.rank() >= 2,
                "Input rank must be >= 2, for the the batch and channel axes, got {}",
                shape
            );

            let rest_size = shape.dims[2..].iter().copied().product::<Size>();
            let flat_shape = shape![shape[0], shape[1], rest_size];
            let broadcast_shape = shape.keep(1, Size::ONE);

            let flat = graph.view(input, flat_shape);
            let norm_flat = graph.layernorm(flat, 2, epsilon);
            let norm = graph.view(norm_flat, shape);

            let scale_broadcast = graph.view(scale, broadcast_shape.clone());
            let bias_broadcast = graph.view(bias, broadcast_shape);

            let scaled = graph.mul(norm, scale_broadcast);
            let result = graph.add(scaled, bias_broadcast);

            OnnxValue::Value(result)
        }
        "Constant" => {
            let tensor = attrs.take_tensor("value")?;
            let value = define_tensor_data(graph, node.name, tensor, external)?;
            OnnxValue::Value(value)
        }
        "ConstantOfShape" => {
            let shape = inputs.optional(0);

            let shape = match shape {
                None => Shape::SCALAR,
                Some(shape) => shape.as_shape(graph)?,
            };

            let value = match attrs.maybe_take_tensor("value")? {
                None => graph.scalar(0f32),
                Some(tensor) => define_tensor_data(graph, node.name, tensor, external)?,
            };

            // TODO force scalar value? spec is unclear
            assert_eq!(
                graph[value].shape.size(),
                Size::ONE,
                "value must be a one-element tensor"
            );

            let scalar = graph.view(value, Shape::SCALAR);
            let result = graph.broadcast(scalar, shape);
            OnnxValue::Value(result)
        }
        "Cast" => {
            let input = inputs.required(0)?;
            let dtype = DataType::try_from(attrs.take_int("to")? as i32).expect("Invalid data type");
            let dtype = resolve_dtype(dtype)?;

            match input {
                &OnnxValue::Value(value) => {
                    OnnxValue::Value(graph.unary(UnaryOp::ValueCast(dtype), value))
                }
                OnnxValue::Size(value) => {
                    // only allow no-op casts for now
                    assert_eq!(dtype, DType::I64);
                    OnnxValue::new_size(value.clone(), graph)
                }
            }
        }
        "Reshape" => {
            let input = inputs.required(0)?;
            let new_shape = inputs.required(1)?.as_signed_shape(graph)?;
            let allow_zero = attrs.maybe_take_bool("allowzero")?.unwrap_or(false);

            let old_shape = input.shape(graph);
            let output_shape = calculate_reshape_output_shape(&old_shape, &new_shape, allow_zero);

            match input {
                &OnnxValue::Value(input) => OnnxValue::Value(graph.view(input, output_shape)),
                OnnxValue::Size(input) => {
                    let result = input.reshape(output_shape.unwrap_fixed("reshape shape").dims.clone());
                    OnnxValue::new_size(result, graph)
                }
            }
        }
        "Expand" => {
            let input = inputs.required(0)?;
            let shape = inputs.required(1)?.as_shape(graph)?;

            // "Expand" is a symmetric broadcast, not just a directional one
            let result_shape = broadcast_shape_symmetric(&input.shape(&graph), &shape);

            match input {
                &OnnxValue::Value(input) => OnnxValue::Value(graph.broadcast(input, result_shape)),
                OnnxValue::Size(input) => {
                    let result_shape = result_shape.unwrap_fixed("expand shape").dims.clone();
                    let result = input.broadcast(result_shape).unwrap().to_shared();
                    OnnxValue::new_size(result, graph)
                }
            }
        }
        "Unsqueeze" => {
            let input = inputs.required(0)?;

            let rel_axes = match inputs.optional(1) {
                Some(rel_axes) => {
                    let shape = rel_axes.as_signed_shape(graph)?;
                    shape.iter().map(|d| d.unwrap_fixed().unwrap()).collect_vec()
                }
                None => attrs.take_ints("axes")?.to_vec(),
            };

            // calculate output shape
            let input_shape = input.shape(graph);

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

            // map value
            match input {
                &OnnxValue::Value(input) => OnnxValue::Value(graph.view(input, output_shape)),
                OnnxValue::Size(input) => {
                    let result_shape = output_shape.unwrap_fixed("unsqueeze shape").dims;
                    let result = input.reshape(result_shape);
                    OnnxValue::new_size(result, graph)
                }
            }
        }
        "Transpose" => {
            let input = inputs.required(0)?;

            let permutation = attrs.take_ints("perm")?;
            let permutation = permutation.iter().map(|&x| x as usize).collect_vec();

            match input {
                &OnnxValue::Value(input) => OnnxValue::Value(graph.permute(input, permutation)),
                OnnxValue::Size(input) => {
                    let result = input.to_shared().permuted_axes(permutation);
                    OnnxValue::new_size(result, graph)
                }
            }
        }
        "Gather" => {
            let input = inputs.required(0)?;
            let indices = inputs.required(1)?;
            let rel_axis = attrs.maybe_take_int("axis")?.unwrap_or(0);

            let input_shape = input.shape(graph);
            let axis = abs_axis(rel_axis, input_shape.rank());

            match input {
                &OnnxValue::Value(input) => {
                    // TODO properly support negative indices, either by remapping here or in the graph operator
                    let result = graph.gather(input, axis, indices.unwrap_value().unwrap());
                    OnnxValue::Value(result)
                }
                OnnxValue::Size(input) => {
                    let indices = graph.as_const(indices.unwrap_value().unwrap()).unwrap();

                    // shape trickery to support multi-dim gathers
                    let indices_flat = indices.reshape(vec![indices.len()]);
                    let result_flat = cpu_gather(input, axis, indices_flat);
                    let mut result_shape = input.shape().to_owned();
                    result_shape.splice(axis..axis + 1, indices.shape().iter().copied());
                    let result = result_flat.reshape(result_shape);

                    OnnxValue::new_size(result, graph)
                }
            }
        }
        "Slice" => {
            let get =
                |inputs: &mut Inputs, attrs: &mut Attributes, index: usize, name: &str| -> OnnxResult<_> {
                    match inputs.optional(index) {
                        Some(value) => {
                            let value = graph.as_const(value.unwrap_value().unwrap()).unwrap();

                            assert_eq!(
                                value.shape().len(),
                                1,
                                "Slice operand {} must be 1D const, got shape {:?}",
                                name,
                                value.shape()
                            );

                            let vec = match value {
                                DTensor::I64(value) => value.iter().copied().collect_vec(),
                                DTensor::I32(value) => value.iter().map(|&x| x as i64).collect_vec(),
                                _ => panic!("Invalid slice operand type {:?}", value.dtype()),
                            };

                            Ok(Some(vec))
                        }
                        None => Ok(attrs.maybe_take_ints(name)?.map(|v| v.to_vec())),
                    }
                };

            let input = inputs.required(0)?;
            let starts = get(inputs, attrs, 1, "starts")?.expect("Missing starts input and attribute");
            let ends = get(inputs, attrs, 2, "ends")?.expect("Missing ends input and attribute");

            let slice_rank = starts.len();
            let axes = get(inputs, attrs, 3, "axes")?.unwrap_or_else(|| (0..slice_rank as i64).collect_vec());
            let steps = get(inputs, attrs, 4, "steps")?.unwrap_or_else(|| vec![1; slice_rank]);

            assert!(
                slice_rank == ends.len() && slice_rank == axes.len() && slice_rank == steps.len(),
                "Inconsistent axes count"
            );
            assert!(
                axes.iter().all_unique(),
                "Slice axis cannot be repeated, got {:?}",
                axes
            );

            let input_shape = input.shape(graph);

            (0..slice_rank).fold(input.clone(), |curr, i| {
                let axis = abs_axis(axes[i], input_shape.rank());
                let axis_size = input_shape[axis].unwrap_fixed("Slice axis size");

                let step = steps[i];
                assert_ne!(step, 0, "Step cannot be 0");

                if step > 0 {
                    let start = abs_axis(starts[i], axis_size);
                    let end = abs_axis(ends[i], axis_size);

                    let range = SliceRange::new(start, end, step as usize);

                    // slice
                    match curr {
                        OnnxValue::Value(curr) => OnnxValue::Value(graph.slice(curr, axis, range)),
                        OnnxValue::Size(curr) => OnnxValue::Size(cpu_slice(&curr, axis, range)),
                    }
                } else {
                    // TODO support all negative strides?
                    assert!(
                        starts[i] == -1 && ends[i] == i64::MIN && steps[i] == -1,
                        "Only simple flip negative stride supported for now"
                    );

                    // flip
                    match curr {
                        OnnxValue::Value(curr) => OnnxValue::Value(graph.flip(curr, axis)),
                        OnnxValue::Size(curr) => OnnxValue::Size(cpu_flip(&curr, axis)),
                    }
                }
            })
        }
        "Concat" => {
            let inputs = inputs.take_all_variadic();
            assert!(!inputs.is_empty(), "Must concatenate at least one value");

            let rank = inputs[0].shape(graph).rank();

            let rel_axis = attrs.take_int("axis")?;
            let axis = abs_axis(rel_axis, rank);

            let any_shape = inputs.iter().any(|x| matches!(x, OnnxValue::Size(_)));

            if any_shape {
                let tensors: Vec<_> = inputs.iter().map(|x| x.as_size(graph)).try_collect()?;
                let views: Vec<_> = tensors.iter().map(|x| x.view()).collect();
                let result = ndarray::concatenate(Axis(axis), &views).unwrap().into_shared();
                OnnxValue::new_size(result, graph)
            } else {
                let inputs = inputs.iter().map(|v| v.unwrap_value().unwrap()).collect();
                let result = graph.concat(inputs, axis, None, None);
                OnnxValue::Value(result)
            }
        }
        "Pad" => {
            // operands
            let input = inputs.required(0)?.unwrap_value().unwrap();
            let pads = inputs.required(1)?.unwrap_value().unwrap();
            let constant_value = inputs.optional(2);
            let axes = inputs.optional(3);
            let mode = attrs.maybe_take_string("mode")?.unwrap_or("constant");

            // map operands
            let input_shape = &graph[input].shape.clone();

            let constant_value = constant_value
                .map(|v| graph.as_single_const(v.unwrap_value().unwrap()).unwrap().unwrap_f32().unwrap())
                .unwrap_or(0.0);

            let axes = match axes {
                Some(axes) => {
                    let axes = axes.as_signed_shape(graph)?;
                    axes.iter().map(|&i| abs_axis(i.unwrap_fixed().unwrap(), input_shape.rank())).collect_vec()
                }
                None => (0..input_shape.rank()).collect_vec(),
            };

            let pads = graph.as_const(pads).unwrap();
            let pads = pads.unwrap_i64().unwrap();
            assert_eq!(pads.shape(), &[axes.len() * 2], "Pads and axes shape mismatch");
            let pads = pads.iter().copied().collect_vec();

            assert_eq!(mode, "constant", "Only 'constant' pad mode supported");

            let constant = graph.scalar(constant_value);

            // TODO consider adding a real pad operation instead of this concat workaround
            let output = axes.iter().fold(input, |acc, &axis| {
                let acc_shape = graph[acc].shape.clone();

                let pad_left = pads[axis];
                let pad_right = pads[axes.len() + axis];
                assert!(pad_left >= 0 && pad_right >= 0, "Pad values cannot be negative");

                let blocks = vec![
                    graph.broadcast(constant, acc_shape.replace(axis, shape![pad_left as usize])),
                    acc,
                    graph.broadcast(constant, acc_shape.replace(axis, shape![pad_right as usize])),
                ];
                graph.concat(blocks, axis, None, None)
            });

            OnnxValue::Value(output)
        }
        "Shape" => {
            let input = inputs.required(0)?;
            let shape = input.shape(graph);
            let dims = shape.dims.iter().map(|&d| SignedSize::from_size(d).unwrap()).collect_vec();
            OnnxValue::new_size(Tensor::from_shape_vec(vec![dims.len()], dims).unwrap(), graph)
        }
        "Identity" => {
            let input = inputs.required(0)?;
            input.clone()
        }
        "Softmax" => {
            let input = inputs.required(0)?.unwrap_value().unwrap();

            let shape = graph[input].shape.clone();
            let axis = attrs.maybe_take_int("axis")?.unwrap_or(-1);
            let axis = abs_axis(axis, shape.rank());

            OnnxValue::Value(graph.softmax(input, axis))
        }
        "ReduceSum" | "ReduceMean" | "ReduceProd" | "ReduceMin" | "ReduceMax" => {
            let op = match node.op_type {
                "ReduceSum" => ReduceOp::Sum,
                "ReduceMean" => ReduceOp::Mean,
                "ReduceProd" => ReduceOp::Prod,
                "ReduceMin" => ReduceOp::Min,
                "ReduceMax" => ReduceOp::Max,
                _ => unreachable!(),
            };

            let input = inputs.required(0)?.unwrap_value().unwrap();
            let input_shape = graph[input].shape.clone();

            let axes = attrs.maybe_take_ints("axes")?.map_or_else(
                || (0..input_shape.rank()).collect_vec(),
                |axes| axes.iter().map(|&a| abs_axis(a, input_shape.rank())).collect_vec(),
            );
            let keep_dims = attrs.maybe_take_int("keepdims")?.unwrap_or(1) != 0;

            let result_shape = if keep_dims {
                input_shape.replace_all(&axes, shape![1])
            } else {
                input_shape
            };

            let result = graph.reduce(input, axes, op);
            let result_shaped = graph.view(result, result_shape);

            OnnxValue::Value(result_shaped)
        }
        "Resize" => {
            // operands
            let input = inputs.required(0)?;
            let roi = inputs.optional(1);
            let scales = inputs.optional(2).map(|v| v.unwrap_value().unwrap());
            let sizes = inputs.optional(3);

            let _antialias = attrs.maybe_take_bool("antialias")?.unwrap_or(false);
            let axes = attrs.maybe_take_ints("axes")?;
            let _coordinate_transformation_mode = attrs
                .maybe_take_string("coordinate_transformation_mode")?
                .unwrap_or("half_pixel");
            let _cubic_coeff_a = attrs.take_float("cubic_coeff_a")?;
            let _exclude_outside = attrs.maybe_take_int("exclude_outside")?.unwrap_or(0);
            let _extrapolation_value = attrs.maybe_take_float("extrapolation_value")?.unwrap_or(0.0);
            let keep_aspect_ratio_policy = attrs
                .maybe_take_string("keep_aspect_ratio_policy")?
                .unwrap_or("stretch")
                .to_owned();
            let mode = attrs.maybe_take_string("mode")?.unwrap_or("nearest").to_owned();
            let nearest_mode = attrs
                .maybe_take_string("nearest_mode")?
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

            let input_tensor = input.unwrap_value().unwrap();
            let input_shape = &graph[input_tensor].shape;
            let rank = input_shape.rank();

            assert_eq!(
                scales.shape(),
                &[rank],
                "Scales must be a vector with length the input rank"
            );

            let result = scales
                .unwrap_f32()
                .unwrap()
                .iter()
                .enumerate()
                .fold(input_tensor, |acc, (axis, &scale_f)| {
                    let scale = scale_f as usize;
                    assert_eq!(scale as f32, scale_f, "Only integer scales supported, got {:?}", scales);
                    graph.repeat_interleave(acc, axis, Size::fixed(scale))
                });

            OnnxValue::Value(result)
        }
        _ => {
            return Err(OnnxError::UnsupportedOperation(node.to_owned()));
        }
    };

    Ok(result)
}

fn define_tensor_data(
    graph: &mut Graph,
    name: &str,
    tensor: &TensorProto,
    external: &mut dyn ExternalDataLoader,
) -> OnnxResult<Value> {
    let data_location = DataLocation::try_from(tensor.data_location).expect("Illegal data_location");

    // figure out the shape and type
    let dims = tensor.dims.iter().map(|&d| Size::fixed(d as usize)).collect_vec();
    let shape = Shape::new(dims);
    let size = shape.size().unwrap_fixed("Data tensor shape must be fixed");
    let data_type = DataType::try_from(tensor.data_type).expect("Illegal data type");

    let dtype = match data_type {
        DataType::Float => DType::F32,
        DataType::Uint8 => DType::U8,
        DataType::Int8 => DType::I8,
        DataType::Uint16 => DType::U16,
        DataType::Int16 => DType::I16,
        DataType::Int32 => DType::I32,
        DataType::Int64 => DType::I64,
        DataType::Double => DType::F64,
        DataType::Uint32 => DType::U32,
        DataType::Uint64 => DType::U64,
        _ => panic!("Unsupported constant type {:?} {} in {}", data_type, tensor.data_type, name),
    };

    let length_guess = size * dtype.size().bytes();

    // load the data
    let raw_data_slot;
    let raw_data = match data_location {
        DataLocation::Default => {
            // just use the built-in external data
            &tensor.raw_data
        }
        DataLocation::External => {
            // collect external data properties
            let mut location: Option<&str> = None;
            let mut offset: usize = 0;
            let mut length: Option<usize> = None;

            for entry in &tensor.external_data {
                let key: &str = &entry.key;
                let value: &str = &entry.value;

                match key {
                    "location" => location = Some(value),
                    "offset" => offset = value.parse().unwrap(),
                    "length" => length = Some(value.parse().unwrap()),
                    "hash" => {}
                    _ => panic!("Invalid external_data key: {} (value {})", key, value),
                }
            }

            if let Some(length) = length {
            assert_eq!(length, length_guess, "External data length mismatch");
        }

        // try loading from external source
        let location = location.expect("External data must have a location");
        raw_data_slot = external.load_external_data(&PathBuf::from(location), offset, length, length_guess)?;

            if let Some(length) = length {
                assert_eq!(raw_data_slot.len(), length, "Raw data length mismatch");
            }

            &raw_data_slot
        }
    };

    macro_rules! read_type {
        (graph, $T:ty, $data:ident, None) => {{
            let data: Vec<$T> = if tensor.$data.is_empty() {
                raw_data.iter().map(|&x| x as $T).collect()
            } else {
                tensor.$data.iter().map(|&x| x as $T).collect()
            };
            graph.constant::<$T>(shape, data)
        }};
        (graph, $T:ty, $data:ident, $read:ident) => {{
            let data: Vec<$T> = if tensor.$data.is_empty() {
                let mut data = vec![Default::default(); size];
                LittleEndian::$read(raw_data, &mut data);
                data
            } else {
                tensor.$data.iter().map(|&x| x as $T).collect()
            };
            graph.constant::<$T>(shape, data)
        }};
    }

    // careful, this stuff is pretty weirdly mapped, see the TensorProto docs
    let value = match dtype {
        DType::F32 => read_type!(graph, f32, float_data, read_f32_into),
        DType::F64 => read_type!(graph, f64, double_data, read_f64_into),
        DType::I8 => read_type!(graph, i8, int32_data, None),
        DType::I16 => read_type!(graph, i16, int32_data, read_i16_into),
        DType::I32 => read_type!(graph, i32, int32_data, read_i32_into),
        DType::I64 => read_type!(graph, i64, int64_data, read_i64_into),
        DType::U8 => read_type!(graph, u8, int32_data, None),
        DType::U16 => read_type!(graph, u16, int32_data, read_u16_into),
        DType::U32 => read_type!(graph, u32, uint64_data, read_u32_into),
        DType::U64 => read_type!(graph, u64, uint64_data, read_u64_into),
    };

    Ok(value)
}

fn resolve_tensor_type(ty: &TypeProto) -> OnnxResult<(Shape, DType)> {
    let value = ty.value.as_ref().expect("Value doesn't have type set");
    let result = match value {
        ProtoTypeValue::TensorType(tensor) => {
            let data_type = DataType::try_from(tensor.elem_type).expect("Invalid data type");

            let dims = tensor
                .shape
                .as_ref()
                .expect("Tensor does not have shape set")
                .dim
                .iter()
                .map(resolve_tensor_dim)
                .collect_vec();

            let dtype = resolve_dtype(data_type)?;

            (Shape::new(dims), dtype)
        }
        _ => panic!("Unsupported value kind {:?}", value),
    };
    Ok(result)
}

fn resolve_dtype(data_type: DataType) -> OnnxResult<DType> {
    let dtype = match data_type {
        DataType::Float => DType::F32,
        DataType::Uint8 => DType::U8,
        DataType::Int8 => DType::I8,
        DataType::Uint16 => DType::U16,
        DataType::Int16 => DType::I16,
        DataType::Int32 => DType::I32,
        DataType::Int64 => DType::I64,
        DataType::Uint32 => DType::U32,
        DataType::Uint64 => DType::U64,
        DataType::Undefined | DataType::String | DataType::Bool |
        DataType::Float16 | DataType::Double |
        DataType::Complex64 | DataType::Complex128 |
        DataType::Bfloat16 => return Err(OnnxError::UnsupportedType(data_type)),
    };
    Ok(dtype)
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

fn calculate_reshape_output_shape(old_shape: &Shape, new_shape_raw: &[SignedSize], allow_zero: bool) -> Shape {
    let old_size = old_shape.size();

    let mut new_shape = vec![];
    let mut leftover_index = None;
    let mut leftover_size = old_size;

    for (i, &signed_size) in new_shape_raw.iter().enumerate() {
        let size = match signed_size {
            SignedSize::ZERO => {
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
            SignedSize::NEG_ONE => {
                assert!(
                    leftover_index.is_none(),
                    "Reshape shape can only contain a single -1 value"
                );
                leftover_index = Some(i);
                new_shape.push(Size::ZERO);
                continue;
            }
            signed_size => signed_size
                .to_size()
                .unwrap_or_else(|_| panic!("Reshape size must be positive, 0 or -1, got {:?}", signed_size)),
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

fn eval_binary_op(op: BinaryOp, a: SignedSize, b: SignedSize) -> Option<SignedSize> {
    // TODO min/max?
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => Some(a * b),
        BinaryOp::Div => a.floor_div(b),
        _ => None,
    }
}

fn load_model_proto(buf: &[u8]) -> ModelProto {
    let mut buf: &[u8] = buf;
    ModelProto::decode(&mut buf).unwrap()
}
