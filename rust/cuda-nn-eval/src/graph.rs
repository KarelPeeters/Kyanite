use std::fmt::{Debug, Formatter};
use std::ops::Index;

use crate::util::WrapDebug;

pub struct Graph {
    values: Vec<ValueInfo>,
    inputs: Vec<Value>,
    outputs: Vec<Value>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Value(usize);

#[derive(Debug)]
pub struct ValueInfo {
    /// [n, c, w, h]
    pub shape: [i32; 4],
    pub operation: Operation,
}

#[derive(Debug)]
pub enum Operation {
    Input,
    Constant { data: WrapDebug<Vec<f32>> },

    Flatten { input: Value },

    Conv { input: Value, filter: Value, conv_shape: ConvShape },
    Bias { input: Value, bias: Value },
    Add { left: Value, right: Value },
    Relu { input: Value },
}

#[derive(Debug, Copy, Clone)]
pub struct ConvShape {
    pub input_channels: i32,
    pub output_channels: i32,
    pub kernel_width: i32,
    pub kernel_height: i32,
    pub pad_w: i32,
    pub pad_h: i32,
}

impl Index<Value> for Graph {
    type Output = ValueInfo;

    fn index(&self, value: Value) -> &Self::Output {
        self.check_contains(value);
        &self.values[value.0]
    }
}

impl Graph {
    pub fn empty() ->Self {
        Graph { values: vec![], inputs: vec![], outputs: vec![] }
    }

    fn check_contains(&self, value: Value) {
        assert!(value.0 < self.values.len());
    }

    /// Iterate over the values in this graph, in topological order.
    pub fn values(&self) -> impl Iterator<Item=Value> {
        (0..self.values.len()).map(Value)
    }

    pub fn inputs(&self) -> &Vec<Value> {
        &self.inputs
    }

    pub fn outputs(&self) -> &Vec<Value> {
        &self.outputs
    }

    fn push(&mut self, shape: [i32; 4], operation: Operation) -> Value {
        assert_valid_shape(shape);

        let index = self.values.len();
        self.values.push(ValueInfo { shape, operation });
        Value(index)
    }

    /// Declare a new input value.
    #[must_use]
    pub fn input(&mut self, shape: [i32; 4]) -> Value {
        let value = self.push(shape, Operation::Input);
        self.inputs.push(value);
        value
    }

    /// Declare a new constant.
    #[must_use]
    pub fn constant(&mut self, shape: [i32; 4], data: Vec<f32>) -> Value {
        assert_valid_shape(shape);

        let expected_len = shape.iter().product::<i32>();
        assert_eq!(expected_len, data.len() as i32, "Shape {:?} and data size {} mismatch", shape, data.len());

        self.push(shape, Operation::Constant { data: data.into() })
    }

    /// Flatten a value of shape `[n, c, w, h]` to shape `[n, c * w * h, 1, 1]`;
    #[must_use]
    pub fn flatten(&mut self, input: Value) -> Value {
        let [n, c, h, w] = self[input].shape;
        self.push(
            [n, c * h * w, 1, 1],
            Operation::Flatten { input },
        )
    }

    /// 2D convolution.
    #[must_use]
    pub fn conv(&mut self, input: Value, filter: Value, pad_w: i32, pad_h: i32) -> Value {
        let [n, in_c, in_w, in_h] = self[input].shape;
        let [output_channels, input_channels, kernel_width, kernel_height] = self[filter].shape;

        assert_eq!(1, kernel_width % 2, "Kernel width must be odd, got {}", kernel_width);
        assert_eq!(1, kernel_height % 2, "Kernel height must be odd, got {}", kernel_height);

        assert_eq!(in_c, input_channels, "Input channel mismatch");

        let out_w = in_w - kernel_width + 1 + 2 * pad_w;
        let out_h = in_h - kernel_height + 1 + 2 * pad_h;
        let output_shape = [n, output_channels, out_w, out_h];

        let conv_shape = ConvShape { input_channels, output_channels, kernel_width, kernel_height, pad_w, pad_h };
        self.push(
            output_shape,
            Operation::Conv { input, conv_shape, filter },
        )
    }

    /// Channel-wise bias
    #[must_use]
    pub fn bias(&mut self, input: Value, bias: Value) -> Value {
        let [_, in_c, _, _] = self[input].shape;
        let [bias_n, bias_c, bias_w, bias_h] = self[bias].shape;

        assert_eq!(in_c, bias_c, "Channel mismatch");
        assert_eq!(1, bias_n);
        assert_eq!(1, bias_w);
        assert_eq!(1, bias_h);

        self.push(
            self[input].shape,
            Operation::Bias { input, bias },
        )
    }

    /// Elementwise relu.
    #[must_use]
    pub fn relu(&mut self, input: Value) -> Value {
        self.push(self[input].shape, Operation::Relu { input })
    }

    /// Add two same-size values together.
    #[must_use]
    pub fn add(&mut self, left: Value, right: Value) -> Value {
        let left_shape = self[left].shape;
        let right_shape = self[right].shape;
        assert_eq!(left_shape, right_shape, "Both inputs must have the same shape");
        self.push(left_shape, Operation::Add { left, right })
    }

    /// Register an existing value as an output
    pub fn output(&mut self, value: Value) {
        assert!(!self.outputs.contains(&value), "{:?} already registered as an output!", value);
        self.outputs.push(value);
    }
}

impl Debug for Graph {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let Graph { values, inputs, outputs } = self;

        writeln!(f, "Graph {{")?;

        writeln!(f, "  values: [")?;
        for (i, info) in values.iter().enumerate() {
            writeln!(f, "    {:?} -> {:?},", Value(i), info)?;
        }
        writeln!(f, "  ],")?;

        writeln!(f, "  inputs: {:?},", inputs)?;
        writeln!(f, "  outputs: {:?},", outputs)?;

        writeln!(f, "}}")?;

        Ok(())
    }
}

fn assert_valid_shape(shape: [i32; 4]) {
    assert!(shape.iter().all(|&x| x > 0), "Shape must be positive, got {:?}", shape);
}
