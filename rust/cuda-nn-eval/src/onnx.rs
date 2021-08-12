use std::collections::HashMap;
use std::convert::TryInto;
use std::path::Path;

use byteorder::{ByteOrder, LittleEndian};
use cast_trait::Cast;
use itertools::Itertools;
use prost::Message;

use crate::graph::Graph;
use crate::graph::Value;
use crate::onnx::proto::{AttributeProto, ModelProto, TensorProto, TypeProto};
use crate::onnx::proto::attribute_proto::AttributeType;
use crate::onnx::proto::tensor_proto::DataType;
use crate::onnx::proto::tensor_shape_proto::dimension::Value as ProtoShapeValue;
use crate::onnx::proto::type_proto::Value as ProtoTypeValue;

pub fn load_onnx_graph(path: impl AsRef<Path>, batch_size: i32) -> Graph {
    load_onnx_impl(path.as_ref(), batch_size)
}

mod proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

fn load_onnx_impl(path: &Path, batch_size: i32) -> Graph {
    let model = load_model_proto(path);
    let model_graph = model.graph.unwrap();

    for init in &model_graph.initializer {
        assert_eq!(DataType::Float as i32, init.data_type);
    }

    let init_map: HashMap<&str, &TensorProto> = model_graph.initializer.iter()
        .map(|proto| (&*proto.name, proto))
        .collect();
    let mut node_map: HashMap<&str, Value> = HashMap::new();

    let mut graph = Graph::empty();

    for input in &model_graph.input {
        let shape = resolve_tensor_shape(input.r#type.as_ref().unwrap(), batch_size);
        let value = graph.input(shape);
        assert!(node_map.insert(&input.name, value).is_none());
    }

    for node in &model_graph.node {
        let mut attrs = Attributes::from(&node.attribute);

        let value = match &*node.op_type {
            "Conv" => {
                let input = *node_map.get(&&*node.input[0]).expect("Conv input not found");
                let filter = *init_map.get(&&*node.input[1]).expect("Conv filter not found");
                let bias = node.input.get(2).map(|bias| {
                    *init_map.get(&**bias).expect("Conv bias not found")
                });

                // unwrap_match!(attrs.take_ints(""))
                let g = attrs.take_int("group");
                let [kw, kh] = unwrap_2(attrs.take_ints("kernel_shape"));
                let [ph0, pv0, ph1, pv1] = unwrap_4(attrs.take_ints("pads"));
                let [sw, sh] = unwrap_2(attrs.take_ints("strides"));
                let [dw, dh] = unwrap_2(attrs.take_ints("dilations"));
                let [output_channels, input_channels, kernel_w, kernel_h] = unwrap_4(&filter.dims);

                assert_eq!(1, g);
                assert!(ph0 == ph1 && pv0 == pv1);
                assert!(dw == 1 && dh == 1);
                assert!(sw == 1 && sh == 1);
                assert!(kernel_w == kw && kernel_h == kh);

                let filter = graph.constant(
                    cast_shape([output_channels, input_channels, kernel_w, kernel_h]),
                    get_tensor_f32_data(filter),
                );
                let conv = graph.conv(input, filter, ph0 as i32, pv0 as i32);

                if let Some(bias) = bias {
                    let channels = unwrap_1(&bias.dims);

                    let bias = graph.constant(
                        cast_shape([1, channels, 1, 1]),
                        get_tensor_f32_data(bias),
                    );
                    graph.bias(conv, bias)
                } else {
                    conv
                }
            }
            "Relu" => {
                let input = *node_map.get(&&*node.input[0]).expect("Relu input not found");
                graph.relu(input)
            }
            "Add" => {
                let left = *node_map.get(&&*node.input[0]).expect("Add left input not found");
                let right = *node_map.get(&&*node.input[1]).expect("Add right input not found");
                graph.add(left, right)
            }
            "Flatten" => {
                let axis = attrs.take_int("axis");
                assert_eq!(1, axis, "Only flatten starting from axis 1 supported");

                let input = *node_map.get(&&*node.input[0]).expect("Flatten input not found");
                graph.flatten(input)
            }
            "Gemm" => {
                assert_eq!(3, node.input.len());
                let input = *node_map.get(&&*node.input[0]).expect("Gemm input not found");
                let weight = *init_map.get(&&*node.input[1]).expect("Gemm filter not found");
                let bias = node.input.get(2).map(|bias| {
                    *init_map.get(&**bias).expect("Gemm bias not found")
                });

                let alpha = attrs.take_float("alpha");
                let beta = attrs.take_float("beta");
                let trans_b = attrs.take_int("transB") != 0;

                assert!(trans_b);
                assert_eq!(1.0, alpha);
                assert_eq!(1.0, beta);

                let [output_channels, input_channels] = unwrap_2(&weight.dims);

                let filter = graph.constant(
                    cast_shape([output_channels, input_channels, 1, 1]),
                    get_tensor_f32_data(weight),
                );
                let conv = graph.conv(input, filter, 0, 0);

                if let Some(bias) = bias {
                    let channels = unwrap_1(&bias.dims);

                    let bias = graph.constant(
                        cast_shape([1, channels, 1, 1]),
                        get_tensor_f32_data(bias),
                    );
                    graph.bias(conv, bias)
                } else {
                    conv
                }
            }
            _ => panic!("Unsupported op_type '{}'", node.op_type)
        };

        // register operation output as node
        assert_eq!(1, node.output.len(), "nodes with multiple outputs not supported");
        let output = &*node.output[0];
        assert!(node_map.insert(&output, value).is_none());
        assert!(attrs.is_done(), "Leftover attributes: {:?}", attrs);
    }

    for output in &model_graph.output {
        let output_value = *node_map.get(&&*output.name).expect("output not found");
        graph.output(output_value);
    }

    graph
}

fn resolve_tensor_shape(ty: &TypeProto, batch_size: i32) -> [i32; 4] {
    let value = ty.value.as_ref().expect("Value doesn't have type set");
    match value {
        ProtoTypeValue::TensorType(tensor) => {
            assert_eq!(tensor.elem_type, DataType::Float as i32, "only floats supported for now");
            let shape = tensor.shape.as_ref().expect("Tensor does not have shape set");
            assert_eq!(4, shape.dim.len(), "Unexpected shape length, shape: {:?}", &shape.dim);

            let shape = shape.dim.iter().map(|d| {
                match d.value.as_ref().unwrap() {
                    ProtoShapeValue::DimValue(value) => *value as i32,
                    ProtoShapeValue::DimParam(param) => {
                        assert_eq!("batch_size", param);
                        batch_size as i32
                    }
                }
            }).collect_vec();

            shape.as_slice().try_into().unwrap()
        }
        _ => panic!("Unsupported value kind {:?}", value),
    }
}

fn get_tensor_f32_data(tensor: &TensorProto) -> Vec<f32> {
    assert_eq!(tensor.data_type, DataType::Float as i32, "expected float tensor");
    let expected_len = tensor.dims.iter().product::<i64>() as usize;

    if !tensor.float_data.is_empty() {
        assert_eq!(expected_len, tensor.float_data.len());
        tensor.float_data.clone()
    } else {
        assert_eq!(expected_len * 4, tensor.raw_data.len());

        let mut float_data = vec![0.0; expected_len];
        LittleEndian::read_f32_into(&tensor.raw_data, &mut float_data);
        float_data
    }
}

fn cast_shape(shape: [i64; 4]) -> [i32; 4] {
    shape.cast()
}

#[track_caller]
fn unwrap_1(slice: &[i64]) -> i64 {
    assert_eq!(slice.len(), 1, "Expected 1 elements, got {:?}", slice);
    slice[0]
}

#[track_caller]
fn unwrap_2(slice: &[i64]) -> [i64; 2] {
    assert_eq!(slice.len(), 2, "Expected 2 elements, got {:?}", slice);
    slice.try_into().unwrap()
}

#[track_caller]
fn unwrap_4(slice: &[i64]) -> [i64; 4] {
    assert_eq!(slice.len(), 4, "Expected 4 elements, got {:?}", slice);
    slice.try_into().unwrap()
}

fn load_model_proto(path: &Path) -> ModelProto {
    let bytes = std::fs::read(path)
        .unwrap();

    let mut bytes: &[u8] = &bytes;
    let model = ModelProto::decode(&mut bytes)
        .unwrap();

    model
}

#[derive(Debug)]
struct Attributes<'a> {
    inner: HashMap<&'a str, &'a AttributeProto>,
}

impl<'a> Attributes<'a> {
    pub fn from(vec: &'a Vec<AttributeProto>) -> Self {
        let inner: HashMap<&str, &AttributeProto> = vec
            .iter()
            .map(|a| (&*a.name, a))
            .collect();
        Attributes { inner }
    }

    pub fn take(&mut self, key: &str, ty: AttributeType) -> &AttributeProto {
        let attribute = self.inner.remove(key)
            .unwrap_or_else(|| {
                let available = self.inner.keys().collect_vec();
                panic!("Missing attribute {}, available: {:?}", key, available)
            });
        assert_eq!(ty, attribute.r#type(), "Expected type {:?}", ty);
        attribute
    }

    pub fn take_int(&mut self, key: &str) -> i64 {
        self.take(key, AttributeType::Int).i
    }

    pub fn take_ints(&mut self, key: &str) -> &[i64] {
        &self.take(key, AttributeType::Ints).ints
    }

    pub fn take_float(&mut self, key: &str) -> f32 {
        self.take(key, AttributeType::Float).f
    }

    pub fn is_done(&self) -> bool {
        self.inner.is_empty()
    }
}

