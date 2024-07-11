use std::path::PathBuf;

use byteorder::{ByteOrder, LittleEndian};
use itertools::{Itertools, zip_eq};
use ndarray::{Axis, azip};
use prost::Message;

use crate::cpu::{cpu_flip, cpu_gather, cpu_slice};
use crate::dtype::{DBool, dispatch_dtensor, DScalar, DTensor, DType, IntoDScalar, map_dtensor_pair, Tensor};
use crate::graph::{
    BinaryOp, broadcast_shape_symmetric, broadcast_tensors_symmetric, ReduceOp, SliceRange, UnaryOp, Value,
};
pub use crate::graph::Graph;
use crate::onnx::external_data::ExternalDataLoader;
use crate::onnx::inputs::{Attributes, Inputs};
use crate::onnx::proto::{ModelProto, TensorProto, TypeProto};
use crate::onnx::proto::tensor_proto::DataLocation;
use crate::onnx::proto::tensor_proto::DataType;
use crate::onnx::proto::tensor_shape_proto::dimension;
use crate::onnx::proto::type_proto::Value as ProtoTypeValue;
use crate::onnx::result::{Node, OnnxError, OnnxResult, UnwrapProto};
use crate::onnx::store::Store;
use crate::onnx::typed_value::{OnnxValue, SignedSize};
use crate::shape;
use crate::shape::{DivResult, Shape, Size};

// TODO we should switch to taking an extra `HashMap<String, Size>` parameter,
//   so the user can decide which named axes match to what size or even the batch size

// TODO convert every possible panic to an error (even in the shape classes if possible)
//    things to grep for: unwrap|expect|assert|panic
//    introduce two main error kinds: "bug in file" and "unsupported"

pub type InputShaper = dyn Fn(&[OnnxDimValue], &str, usize) -> Option<Shape>;

#[derive(Debug, Clone)]
pub enum OnnxDimValue {
    Value(i64),
    Param(String),
}

// we use &dyn to avoid duplicate codegen of this large and non-critical function
pub fn graph_from_onnx_bytes(buf: &[u8], external: &mut dyn ExternalDataLoader, input_shaper: &InputShaper) -> OnnxResult<Graph> {
    let model = load_model_proto(buf);
    let model_graph = model.graph.as_ref().unwrap_proto("model.graph")?;

    let mut graph = Graph::new();
    let mut nodes: Store<OnnxValue> = Store::default();

    // load initializer values (similar to constants but defined separately)
    for tensor in &model_graph.initializer {
        let value = define_tensor_data(&mut graph, &tensor.name, tensor, external)?;
        nodes.define(&tensor.name, OnnxValue::Value(value))
    }

    // load inputs
    let mut real_input_index = 0;
    for input in &model_graph.input {
        // initializers are allowed to re-appear in the inputs, so we skip them the second time
        if nodes.contains(&input.name) {
            continue;
        }

        let input_proto = input.r#type.as_ref().unwrap_proto("input.type")?;
        let (shape, dtype) = resolve_tensor_type(input_proto, &input.name, real_input_index, input_shaper)?;
        let value = graph.input(shape, dtype);
        nodes.define(&input.name, OnnxValue::Value(value));

        real_input_index += 1;
    }

    // clear newly defined values so we don't attribute them to the first node
    let _ = graph.take_new_values();

    // load nodes
    for node_proto in &model_graph.node {
        let node = Node {
            name: node_proto.name.as_str(),
            op_type: node_proto.op_type.as_str(),
        };

        let mut attrs = Attributes::from(node, &node_proto.attribute);
        let mut inputs = Inputs::from(node, &node_proto.input, &nodes)?;

        let values: Vec<OnnxValue> = visit_node(&mut graph, external, node, &mut inputs, &mut attrs)?;

        // set debug id for all newly created nodes to the current node name
        for value in graph.take_new_values() {
            graph.set_debug_id(value, node.name.to_owned())
        }

        // check that the value if only a size if necessary
        for value in &values {
            value.assert_valid();
        }

        // check that we used all attributes and inputs
        let leftover_attributes = attrs.leftover();
        if !leftover_attributes.is_empty() {
            return Err(OnnxError::LeftoverAttributes(node.to_owned(), leftover_attributes));
        }
        let leftover_inputs = inputs.leftover();
        if !leftover_inputs.is_empty() {
            return Err(OnnxError::LeftoverInputs(node.to_owned(), leftover_inputs));
        }

        // actually define the result values
        let output_names = &node_proto.output;
        assert_eq!(output_names.len(), values.len(), "Expected {:?} outputs, got {}", output_names, values.len());
        for (name, value) in zip_eq(output_names, values) {
            nodes.define(name, value);
        }
    }

    for output in &model_graph.output {
        let value_or_size = &nodes[output.name.as_str()];
        let value = value_or_size
            .unwrap_value()
            .ok_or(OnnxError::ExpectedNonBatchValue(output.name.clone()))?;
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
) -> OnnxResult<Vec<OnnxValue>> {
    let result_single = match node.op_type {
        "Conv" => {
            let input = inputs.required(0)?.unwrap_value().unwrap();
            let filter = inputs.required(1)?.unwrap_value().unwrap();
            let bias_raw = inputs.optional(2).map(|v| v.unwrap_value().unwrap());

            let groups = attrs.maybe_take_int("group")?.unwrap_or(1);
            let kernel_shape = attrs.take_ints("kernel_shape")?;
            let conv_rank = kernel_shape.len();
            let strides = attrs.maybe_take_ints("strides")?
                .map_or(vec![1; conv_rank], |strides| strides.to_vec());
            let dilations = attrs.maybe_take_ints("dilations")?
                .map_or(vec![1; conv_rank], |strides| strides.to_vec());

            let auto_pad = attrs.maybe_take_string("auto_pad")?;

            let padding = match auto_pad {
                None | Some("NOTSET") => {
                    // custom padding
                    attrs.take_ints("pads")?.to_vec()
                }
                Some("SAME_UPPER") => {
                    // input and output same size, excess on upper side of dim
                    calculate_auto_padding(graph, conv_rank, input, filter, &strides, &dilations, true)?
                }
                Some("SAME_LOWER") => {
                    // input and output same size, excess on lower side of dim
                    calculate_auto_padding(graph, conv_rank, input, filter, &strides, &dilations, false)?
                }
                Some("VALID") => {
                    // no padding
                    vec![0; strides.len()]
                }
                Some(auto_pad) => return Err(OnnxError::InvalidAutoPadValue(node.to_owned(), auto_pad.to_owned()))
            };

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

            let result = match conv_rank {
                1 => {
                    let kernel_size0 = unwrap_1(kernel_shape);
                    let [padding_0, padding_1] = unwrap_2(&padding);
                    let stride = unwrap_1(&strides);
                    let dilation = unwrap_1(&dilations);

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
                    let [padding_y0, padding_x0, padding_y1, padding_x1] = unwrap_4(&padding);
                    let [stride_y, stride_x] = unwrap_2(&strides);
                    let [dilation_y, dilation_x] = unwrap_2(&dilations);

                    let [_, _, kernel_h1, kernel_w1] = filter_shape.unwrap_4();

                    assert!(padding_y0 == padding_y1 && padding_x0 == padding_x1);
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
                    eval_binary_op(op, l, r)
                        .unwrap_or_else(|| panic!("Operation {:?} failed between {:?} and {:?}", op, left, right))
                });

                // the batch size might have cancelled out!
                OnnxValue::new_size(result.into_shared(), graph)
            }
        }
        "Equal" => {
            let left = inputs.required(0)?;
            let right = inputs.required(1)?;

            let result = match (left, right) {
                (&OnnxValue::Value(left), &OnnxValue::Value(right)) => {
                    // subtract and cast to bool
                    // this automatically broadcasts correctly
                    let diff = graph.sub(left, right);
                    graph.unary(UnaryOp::ValueCast(DType::Bool), diff)
                }
                (OnnxValue::Size(left), OnnxValue::Size(right)) => {
                    // broadcast and compare
                    // TODO we consider batch and ints always not-equal, even though they theoretically could be
                    let (left, right) = broadcast_tensors_symmetric(&left, &right);

                    let result = azip!(left, right).map_collect(|l, r| DBool(l == r)).into_shared();
                    graph.constant_tensor(DTensor::Bool(result))
                }
                _ => {
                    // one contains batch, the other doesn't => they can't be equal
                    // return false of the right shape
                    let broadcast_shape = broadcast_shape_symmetric(&left.shape(graph), &right.shape(graph));
                    let scalar = graph.scalar(DBool(false));
                    graph.broadcast(scalar, broadcast_shape)
                }
            };

            OnnxValue::Value(result)
        }
        "Where" => {
            // TODO extend to non-consts and shapes
            let cond = inputs.required(0)?.unwrap_value().unwrap();
            let x = inputs.required(1)?.unwrap_value().unwrap();
            let y = inputs.required(2)?.unwrap_value().unwrap();

            let cond = graph.as_const(cond).unwrap();
            let cond = cond.unwrap_bool().unwrap();
            let x = graph.as_const(x).unwrap();
            let y = graph.as_const(y).unwrap();

            // TODO proper broadcasting
            assert_eq!(cond.shape(), x.shape(), "Where broadcasting not yet implemented");
            assert_eq!(cond.shape(), y.shape(), "Where broadcasting not yet implemented");

            let result = map_dtensor_pair!(x, y, |x, y| {
                azip!(cond, &x, &y)
                    .map_collect(|&DBool(c), &x, &y| if c { x } else { y })
                    .into_shared()
            });

            OnnxValue::Value(graph.constant_tensor(result))
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
                &OnnxValue::Value(input) => OnnxValue::Value(graph.view(input, new_shape)),
                OnnxValue::Size(input) => {
                    OnnxValue::new_size(input.reshape(new_shape.unwrap_fixed("size shape").dims), graph)
                }
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
            let spatial = attrs.maybe_take_int("spatial")?;
            assert!(
                spatial == None || spatial == Some(1),
                "non-spatial cases are not supported and have been deprecated since ONNX version 9"
            );

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
            let data_type = DataType::try_from(attrs.take_int("to")? as i32).expect("Invalid data type");
            let dtype = resolve_dtype(data_type, node.name)?;

            match input {
                &OnnxValue::Value(value) => OnnxValue::Value(graph.unary(UnaryOp::ValueCast(dtype), value)),
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
            let indices_raw = inputs.required(1)?;
            let rel_axis = attrs.maybe_take_int("axis")?.unwrap_or(0);

            let input_shape = input.shape(graph);
            let axis = abs_axis(rel_axis, input_shape.rank());
            let axis_size = input.shape(graph).dims[axis];

            let indices = match indices_raw {
                &OnnxValue::Value(indices) => {
                    match graph.as_const(indices) {
                        Some(indices) => {
                            let dim = axis_size.unwrap_fixed("gather dim size");
                            let indices = dispatch_dtensor!(indices, |T, ft, indices| {
                                // this is super cursed but it seems to work
                                let zero = T::from_dscalar(T::DTYPE.specials().zero).unwrap();
                                let dim = T::from_dscalar(DScalar::U64(dim as u64).value_cast(T::DTYPE)).unwrap();

                                ft(indices.mapv(|x| if x < zero { x + dim } else { x }).into_shared())
                            });
                            OnnxValue::Value(graph.constant_tensor(indices))
                        }
                        // TODO support dynamic negative indices, by properly remapping in the graph
                        //   for now just hope for the best
                        None => OnnxValue::Value(indices),
                    }
                }
                OnnxValue::Size(indices) => {
                    let indices = indices.mapv(|x| {
                        if x.is_neg() {
                            (x + axis_size).expect("gather negative index overflow")
                        } else {
                            x
                        }
                    });
                    OnnxValue::new_size(indices.into_shared(), graph)
                }
            };

            match input {
                &OnnxValue::Value(input) => {
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
            let get = |inputs: &mut Inputs, attrs: &mut Attributes, index: usize, name: &str| -> OnnxResult<_> {
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
        "Split" => {
            // TODO support "num_outputs" and "split" attribute/input
            let input = inputs.required(0)?;
            let shape = input.shape(graph);

            let axis = attrs.take_int("axis")?;
            let axis = abs_axis(axis, shape.rank());

            let num_outputs = 2;
            let size = shape[axis].unwrap_fixed("Split axis length");

            let len_first = (size + num_outputs - 1) / num_outputs;

            let result = match input {
                &OnnxValue::Value(input) => {
                    vec![
                        OnnxValue::Value(graph.slice(input, axis, SliceRange::simple(0, len_first))),
                        OnnxValue::Value(graph.slice(input, axis, SliceRange::simple(len_first, size))),
                    ]
                }
                OnnxValue::Size(input) => {
                    vec![
                        OnnxValue::new_size(input.slice_axis(Axis(axis), ndarray::Slice::from(..len_first)).into_owned().into_shared(), graph),
                        OnnxValue::new_size(input.slice_axis(Axis(axis), ndarray::Slice::from(len_first..)).into_owned().into_shared(), graph),
                    ]
                }
            };

            return Ok(result);
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
                .map(|v| {
                    graph
                        .as_single_const(v.unwrap_value().unwrap())
                        .unwrap()
                        .unwrap_f32()
                        .unwrap()
                })
                .unwrap_or(0.0);

            let axes = match axes {
                Some(axes) => {
                    let axes = axes.as_signed_shape(graph)?;
                    axes.iter()
                        .map(|&i| abs_axis(i.unwrap_fixed().unwrap(), input_shape.rank()))
                        .collect_vec()
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
            let dims = shape
                .dims
                .iter()
                .map(|&d| SignedSize::from_size(d).unwrap())
                .collect_vec();
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
                input_shape.replace_all(&axes, shape![])
            };

            let result = graph.reduce(input, axes, op);
            let result_shaped = graph.view(result, result_shape);

            OnnxValue::Value(result_shaped)
        }
        "MaxPool" | "AveragePool" => {
            let op = match node.op_type {
                "MaxPool" => ReduceOp::Max,
                "AveragePool" => ReduceOp::Mean,
                _ => unreachable!(),
            };

            let input = inputs.required(0)?.unwrap_value().unwrap();

            let strides = attrs.take_ints("strides")?;
            let kernel_shape = attrs.take_ints("kernel_shape")?;
            let pads = attrs.take_ints("pads")?;
            let ceil_mode = attrs.maybe_take_int("ceil_mode")?.unwrap_or(0) != 0;
            let auto_pad = attrs.maybe_take_string("auto_pad")?;

            assert_eq!(strides, kernel_shape, "Real strides not supported yet");
            assert_eq!(pads, &vec![0; pads.len()], "Padding not supported yet");
            assert!(matches!(auto_pad, None | Some("NOTSET")), "Auto padding not supported yet");

            // max pool the last N dimensions:
            // split each pooled axis into (input_size/kernel_size, kernel_size), then max pool over all kernel sizes
            let raw_input_shape = &graph[input].shape;
            let input_rank = raw_input_shape.rank();
            let kernel_rank = kernel_shape.len();

            let kept_rank = input_rank - kernel_rank;
            let (batch_shape, active_shape) = raw_input_shape.split(kept_rank);

            // calculate padding and reshaping
            let mut pad_amounts = vec![(0, 0); kept_rank];
            let mut reshape = batch_shape.dims.clone();
            let mut pooled_dims = vec![];
            let mut slices = vec![None; kept_rank];

            for i in 0..kernel_rank {
                let kernel_size = kernel_shape[i] as usize;
                let input_size = active_shape.dims[i];

                let div_rem = input_size.div_rem(kernel_size);
                let (left, pad, slice) = match div_rem {
                    DivResult::Exact(left) => {
                        (left, (0, 0), None)
                    }
                    DivResult::Remainder(rem) => {
                        if ceil_mode {
                            let pad = kernel_size - rem;
                            let left = ((input_size + pad).unwrap() / kernel_size).unwrap();
                            (left, (0, pad), None)
                        } else {
                            let left = ((input_size - rem).unwrap() / kernel_size).unwrap();
                            let end = left.unwrap_fixed("pool dim size") * kernel_size;
                            let slice = SliceRange::new(0, end, 1);
                            (left, (0, 0), Some(slice))
                        }
                    },
                    DivResult::Impossible => {
                        return Err(OnnxError::NonDividingPooling(node.to_owned(), raw_input_shape.clone(), kernel_shape.to_vec()));
                    }
                };

                pad_amounts.push(pad);
                reshape.push(left);
                pooled_dims.push(reshape.len());
                reshape.push(Size::fixed(kernel_size));
                slices.push(slice);
            }
            let reshape = Shape::new(reshape);

            let pad_value = op.identity(graph[input].dtype);

            // add to graph
            let pad_value = graph.scalar_dyn(pad_value);
            let padded = graph.pad(input, &pad_amounts, pad_value);
            let sliced = slices.iter().enumerate().fold(padded, |a, (i, &s)| {
                if let Some(s) = s {
                    graph.slice(a, i, s)
                } else {
                    a
                }
            });
            let reshaped = graph.view(sliced, reshape);
            let result = graph.reduce(reshaped, pooled_dims, op);

            OnnxValue::Value(result)
        }
        "GlobalMaxPool" | "GlobalAveragePool" => {
            let op = match node.op_type {
                "GlobalMaxPool" => ReduceOp::Max,
                "GlobalAveragePool" => ReduceOp::Mean,
                _ => unreachable!(),
            };

            let input = inputs.required(0)?.unwrap_value().unwrap();

            // pool the channel dimension
            let shape = &graph[input].shape;
            if shape.rank() < 2 {
                return Err(OnnxError::UnsupportedShape(node.to_owned(), shape.to_string()));
            }

            let axes = (2..shape.rank()).collect_vec();
            let result = graph.reduce(input, axes, op);

            OnnxValue::Value(result)
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

    Ok(vec![result_single])
}

fn calculate_auto_padding(graph: &Graph, conv_rank: usize, input: Value, filter: Value, strides: &[i64], dilations: &[i64], up: bool) -> OnnxResult<Vec<i64>> {
    let (_, input_spatial_dims) = graph[input].shape.split(2);
    let input_spatial_dims = input_spatial_dims.unwrap_fixed("conv input spatial dims");

    let (_, filter_spatial_dims) = graph[filter].shape.split(2);
    let filter_spatial_dims = filter_spatial_dims.unwrap_fixed("conv filter spatial dims");

    let mut result = vec![];
    for i in 0..conv_rank {
        let (low, high) = split_padding(input_spatial_dims.dims[i] as i64, filter_spatial_dims.dims[i] as i64, strides[i], dilations[i], up);
        result.push(low);
        result.push(high);
    }
    Ok(result)
}

fn split_padding(i: i64, f: i64, s: i64, d: i64, up: bool) -> (i64, i64) {
    let total = (i - 1) * s + 1 + d * (f - 1) - i;

    let min = total / 2;
    let max = total - min;

    if up {
        (min, max)
    } else {
        (max, min)
    }
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
    let dtype = resolve_dtype(data_type, name)?;

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
        DType::Bool => {
            let data: Vec<DBool> = if tensor.int32_data.is_empty() {
                raw_data.iter().map(|&x| DBool(x != 0)).collect_vec()
            } else {
                tensor.int32_data.iter().map(|&x| DBool(x != 0)).collect()
            };
            graph.constant::<DBool>(shape, data)
        }
    };

    Ok(value)
}

fn resolve_tensor_type(ty: &TypeProto, name: &str, index: usize, input_shaper: &InputShaper) -> OnnxResult<(Shape, DType)> {
    let value = ty.value.as_ref().expect("Value doesn't have type set");
    let result = match value {
        ProtoTypeValue::TensorType(tensor) => {
            let data_type = DataType::try_from(tensor.elem_type).expect("Invalid data type");
            let dtype = resolve_dtype(data_type, name)?;

            let dims = tensor
                .shape
                .as_ref()
                .expect("Tensor does not have shape set")
                .dim
                .iter()
                .map(|d| match *d.value.as_ref().expect("Missing value for dimension") {
                    dimension::Value::DimValue(value) => OnnxDimValue::Value(value),
                    dimension::Value::DimParam(ref param) => OnnxDimValue::Param(param.clone()),
                })
                .collect_vec();

            let shape = input_shaper(&dims, name, index).ok_or_else(|| OnnxError::FailedToShapeInput(dims, name.to_owned(), index))?;

            (shape, dtype)
        }
        _ => panic!("Unsupported value kind {:?}", value),
    };
    Ok(result)
}

fn resolve_dtype(data_type: DataType, node: &str) -> OnnxResult<DType> {
    let dtype = match data_type {
        DataType::Float => DType::F32,
        DataType::Double => DType::F64,
        DataType::Uint8 => DType::U8,
        DataType::Int8 => DType::I8,
        DataType::Uint16 => DType::U16,
        DataType::Int16 => DType::I16,
        DataType::Int32 => DType::I32,
        DataType::Int64 => DType::I64,
        DataType::Uint32 => DType::U32,
        DataType::Uint64 => DType::U64,
        DataType::Bool => DType::Bool,
        DataType::Undefined
        | DataType::String
        | DataType::Complex64
        | DataType::Complex128
        | DataType::Float16
        | DataType::Bfloat16
        | DataType::Float8e4m3fn
        | DataType::Float8e4m3fnuz
        | DataType::Float8e5m2
        | DataType::Float8e5m2fnuz => return Err(OnnxError::UnsupportedType(node.to_owned(), data_type)),
    };
    Ok(dtype)
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
