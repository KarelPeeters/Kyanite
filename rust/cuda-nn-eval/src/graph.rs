use std::ops::Index;
use std::fmt::{Formatter, Debug};

#[derive(Default)]
pub struct Graph {
    values: Vec<ValueInfo>,
    inputs: Vec<Value>,
    outputs: Vec<Value>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[must_use]
pub struct Value(usize);

#[derive(Debug)]
pub struct ValueInfo {
    pub shape: [i32; 4],
    pub operation: Operation,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Operation {
    Input,
    Conv { input: Value, output_channels: i32, kernel_size: i32, padding: i32, flat_weights: bool },
    Bias { input: Value, channels: i32 },
    Add { left: Value, right: Value },
    Relu { input: Value },
}

impl Index<Value> for Graph {
    type Output = ValueInfo;

    fn index(&self, value: Value) -> &Self::Output {
        self.check_contains(value);
        &self.values[value.0]
    }
}

impl Graph {
    fn check_contains(&self, value: Value) {
        assert!(value.0 < self.values.len());
    }

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
        assert!(shape.iter().all(|&x| x > 0), "shape must be positive, got {:?}", shape);

        let index = self.values.len();
        self.values.push(ValueInfo { shape, operation });
        Value(index)
    }

    pub fn input(&mut self, shape: [i32; 4]) -> Value {
        let value = self.push(shape, Operation::Input);
        self.inputs.push(value);
        value
    }

    pub fn conv_bias_impl(&mut self, input: Value, output_channels: i32, kernel_size: i32, padding: i32, flat_weights: bool) -> Value {
        let [n, _, w, h] = self[input].shape;
        assert_eq!(1, kernel_size % 2, "kernel size must be odd, got {}", kernel_size);

        let output_w = w - kernel_size + 1 + 2 * padding;
        let output_h = h - kernel_size + 1 + 2 * padding;
        let shape = [n, output_channels, output_w, output_h];

        let conv_output = self.push(shape, Operation::Conv {
            input,
            output_channels,
            kernel_size,
            padding,
            flat_weights,
        });
        let bias_output = self.push(shape, Operation::Bias {
            input: conv_output,
            channels: output_channels,
        });
        bias_output
    }

    /// 2D convolution with padding, followed by per-channel bias.
    pub fn conv_bias(&mut self, input: Value, output_channels: i32, kernel: i32, padding: i32) -> Value {
        self.conv_bias_impl(input, output_channels, kernel, padding, false)
    }

    /// Flatten the last 3 dimensions, followed by a fully connected layer, followed by bias.
    pub fn flatten_linear_bias(&mut self, input: Value, output_size: i32) -> Value {
        let input_shape = self[input].shape;
        let [_, _, w, h] = input_shape;
        assert_eq!(w, h, "Only supports square inputs tensors, got {:?}", input_shape);

        self.conv_bias_impl(input, output_size, w, 0, true)
    }

    /// Elementwise relu.
    pub fn relu(&mut self, input: Value) -> Value {
        self.push(self[input].shape, Operation::Relu { input })
    }

    /// Add two same-size values together.
    /// TODO for now the "resnet highway" should be on the left.
    pub fn add(&mut self, left: Value, right: Value) -> Value {
        let left_shape = self[left].shape;
        let right_shape = self[right].shape;
        assert_eq!(left_shape, right_shape, "Both inputs must have the same shape");
        self.push(left_shape, Operation::Add { left, right })
    }

    /// Register the value as an output
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