use std::convert::TryInto;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Deref, Index};

use itertools::{Itertools, zip_eq};

use crate::shape;
use crate::shape::{Shape, Size};

#[derive(Clone)]
pub struct Graph {
    values: Vec<ValueInfo>,
    inputs: Vec<Value>,
    outputs: Vec<Value>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Value(usize);

#[derive(Debug, Clone)]
pub struct ValueInfo {
    pub shape: Shape,
    pub operation: Operation,
}

/// Wrapper type that prevents the Debug output from getting too large.
#[derive(Clone)]
pub struct ConstantData(Vec<f32>);

/// The core set of operations.
/// Many other operations can be implemented as combinations of these operators, a few examples:
/// * `ReLU` -> `Clip`
/// * `BatchNorm` -> `Bias,Conv,Bias,Conv` (can be fused into adjacent conv too)
/// * `Gemm` -> `Conv,Bias`
#[derive(Debug, Clone)]
pub enum Operation {
    /// A runtime-variable input.
    Input { index: usize },
    /// A constant build into the network.
    Constant { data: ConstantData },

    /// View a value as a different shape.
    View { input: Value },
    /// Change the order of axis in the shape.
    Permute { input: Value, permutation: Vec<usize> },
    /// Slice the last three axis of a value, each with range `start[i]..end[i]`
    Slice { input: Value, axis: usize, start: usize, end: usize },
    /// Gather values from `input` at the indices in `index` on the given axis.
    Gather { input: Value, axis: usize, indices: Value },

    /// Concatenate values along an axis.
    Concat { inputs: Vec<Value>, axis: usize },

    /// The standard convolution operator.
    Conv { input: Value, filter: Value, details: ConvDetails },
    /// Batched matrix multiply.
    MatMul { left: Value, right: Value },

    /// Elementwise add two values, with broadcasting on the right.
    Add { left: Value, right: Value, subtract: bool },
    /// Elementwise multiply two values, with broadcasting on the right value.
    Mul { left: Value, right: Value },

    /// Elementwise clip a value.
    Clamp { input: Value, min: f32, max: f32 },
}

impl Operation {
    pub fn inputs(&self) -> Vec<Value> {
        match self {
            Operation::Input { index: _ } => vec![],
            Operation::Constant { data: _ } => vec![],
            &Operation::View { input } => vec![input],
            &Operation::Permute { input, permutation: _ } => vec![input],
            &Operation::Slice { input, axis: _, start: _, end: _ } => vec![input],
            &Operation::Gather { input, axis: _, indices } => vec![input, indices],
            Operation::Concat { inputs, axis: _ } => inputs.clone(),
            &Operation::Conv { input, filter, details: _ } => vec![input, filter],
            &Operation::MatMul { left, right } => vec![left, right],
            &Operation::Add { left, right, subtract: _ } => vec![left, right],
            &Operation::Mul { left, right } => vec![left, right],
            &Operation::Clamp { input, min: _, max: _ } => vec![input],
        }
    }

    pub(crate) fn clone_map_inputs(&self, mut f: impl FnMut(Value) -> Value) -> Operation {
        match self {
            &Operation::Input { index } =>
                Operation::Input { index },
            Operation::Constant { data } =>
                Operation::Constant { data: data.clone() },
            &Operation::View { input } =>
                Operation::View { input: f(input) },
            &Operation::Permute { input, ref permutation } =>
                Operation::Permute { input: f(input), permutation: permutation.clone() },
            &Operation::Slice { input, axis, start, end } =>
                Operation::Slice { input: f(input), axis, start, end },
            &Operation::Gather { input, axis, indices } =>
                Operation::Gather { input: f(input), axis, indices: f(indices) },
            &Operation::Concat { ref inputs, axis } =>
                Operation::Concat { inputs: inputs.iter().copied().map(f).collect(), axis },
            &Operation::Conv { input, filter, details: conv_shape } =>
                Operation::Conv { input: f(input), filter: f(filter), details: conv_shape },
            &Operation::MatMul { left, right } =>
                Operation::MatMul { left: f(left), right: f(right) },
            &Operation::Add { left, right, subtract } =>
                Operation::Add { left: f(left), right: f(right), subtract },
            &Operation::Mul { left, right } =>
                Operation::Mul { left: f(left), right: f(right) },
            &Operation::Clamp { input, min, max } =>
                Operation::Clamp { input: f(input), min, max },
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ConvDetails {
    pub batch_size: Size,

    pub input_channels: usize,
    pub output_channels: usize,

    pub input_h: usize,
    pub input_w: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub padding_y: usize,
    pub padding_x: usize,
    pub output_h: usize,
    pub output_w: usize,
}

impl ConvDetails {
    pub fn input_shape(&self) -> Shape {
        shape![self.batch_size, self.input_channels, self.input_h, self.input_w]
    }

    pub fn output_shape(&self) -> Shape {
        shape![self.batch_size, self.output_channels, self.output_h, self.output_w]
    }

    pub fn keeps_spatial_shape(&self) -> bool {
        (self.input_h == self.output_h) && (self.input_w == self.output_w)
    }

    pub fn has_padding() {}

    pub fn kernel_shape(&self) -> [usize; 4] {
        [self.output_channels, self.input_channels, self.kernel_h, self.kernel_w]
    }
}

impl Index<Value> for Graph {
    type Output = ValueInfo;

    fn index(&self, value: Value) -> &Self::Output {
        self.check_contains(value);
        &self.values[value.0]
    }
}

impl Graph {
    pub fn new() -> Self {
        Graph { values: vec![], inputs: vec![], outputs: vec![] }
    }

    fn check_contains(&self, value: Value) {
        assert!(value.0 < self.values.len());
    }

    fn check_broadcast(&self, left: Value, right: Value) -> Shape {
        let left_shape = &self[left].shape;
        let right_shape = &self[right].shape;
        assert_eq!(
            left_shape.rank(), right_shape.rank(),
            "Both inputs must have the same rank, got {:?} and {:?}",
            left_shape, right_shape
        );

        for (&l, &r) in zip_eq(&left_shape.dims, &right_shape.dims) {
            assert!(l == r || r == Size::ONE, "Cannot broadcast shape {:?} to {:?}", right_shape, left_shape);
        }

        left_shape.clone()
    }

    /// Iterate over the values in this graph, in topological order,
    /// which means that nodes will only be visited after all of their inputs have been visited.
    pub fn values(&self) -> impl Iterator<Item=Value> {
        (0..self.values.len()).map(Value)
    }

    pub fn inputs(&self) -> &[Value] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[Value] {
        &self.outputs
    }

    pub fn outputs_mut(&mut self) -> &mut Vec<Value> {
        &mut self.outputs
    }

    pub fn as_const(&self, value: Value) -> Option<&[f32]> {
        if let Operation::Constant { data } = &self[value].operation {
            Some(data)
        } else {
            None
        }
    }

    pub fn is_all_zero(&self, value: Value) -> bool {
        self.as_const(value).map_or(false, |x| x.iter().all(|&x| x == 0.0))
    }

    pub fn is_all_one(&self, value: Value) -> bool {
        self.as_const(value).map_or(false, |x| x.iter().all(|&x| x == 1.0))
    }

    #[must_use]
    pub(crate) fn push(&mut self, shape: Shape, operation: Operation) -> Value {
        let index = self.values.len();
        self.values.push(ValueInfo { shape, operation });
        Value(index)
    }

    /// Declare a new input value.
    #[must_use]
    pub fn input(&mut self, shape: Shape) -> Value {
        let index = self.inputs.len();
        let value = self.push(shape, Operation::Input { index });
        self.inputs.push(value);
        value
    }

    /// Declare a new constant.
    #[must_use]
    pub fn constant(&mut self, shape: Shape, data: Vec<f32>) -> Value {
        let expected_len = shape.unwrap_fixed("Constant shape must be fixed").size();
        assert_eq!(expected_len, data.len() as usize, "Shape {:?} and data size {} mismatch", shape, data.len());

        self.push(shape, Operation::Constant { data: ConstantData(data) })
    }

    /// View an existing value as a new shape.
    #[must_use]
    pub fn view(&mut self, input: Value, new_shape: Shape) -> Value {
        let old_shape = &self[input].shape;
        if &new_shape == old_shape {
            return input;
        }

        assert_eq!(
            old_shape.size(), new_shape.size(),
            "New shape {:?} must have the same size as old shape {:?}",
            new_shape, old_shape,
        );

        self.push(new_shape, Operation::View { input })
    }

    /// View a value with a flattened shape.
    /// All axis starting from `start_axis` inclusive are flattened into a single axis.
    #[must_use]
    pub fn flatten(&mut self, input: Value, start_axis: usize) -> Value {
        let old_shape = &self[input].shape;
        assert!(
            old_shape.rank() >= start_axis,
            "Input rank {} to low for start axis {}", old_shape.rank(), start_axis
        );

        let new_shape = if start_axis == 0 {
            shape![1, old_shape.size()]
        } else {
            let kept_dims = &old_shape.dims[..start_axis];
            let flat_size = old_shape.dims[start_axis..].iter().copied().product();

            Shape::new([kept_dims, &[flat_size]].concat())
        };

        self.view(input, new_shape)
    }

    /// Change the order of axis in the shape.
    #[must_use]
    pub fn permute(&mut self, input: Value, permutation: Vec<usize>) -> Value {
        let input_shape = &self[input].shape;

        assert_eq!(permutation.len(), input_shape.rank(), "Permutation rank must match input shape, got {:?} and {:?}", permutation, input_shape);
        assert!(permutation.iter().all_unique(), "Permutation cannot contain repeated axis, got {:?}", permutation);
        assert!(permutation.iter().all(|&i| i < input_shape.rank()), "Permutation axis out of bounds, got {:?}", permutation);

        let result_dims = permutation.iter()
            .map(|&i| input_shape[i])
            .collect_vec();
        let result_shape = Shape::new(result_dims);

        self.push(result_shape, Operation::Permute { input, permutation })
    }

    /// View part of an existing value.
    #[must_use]
    pub fn slice(&mut self, input: Value, axis: usize, start: usize, end: usize) -> Value {
        let old_shape = &self[input].shape;

        assert!(
            axis < old_shape.rank(),
            "Input rank {} too low for axis {}", old_shape.rank(), axis
        );

        let mut new_shape = old_shape.clone();
        let dim = new_shape[axis].unwrap_fixed_mut("Slice axis size");

        if start == 0 && end == *dim {
            return input;
        }

        assert!(
            start < *dim && end <= *dim,
            "Slice range {}..{} out of bounds for axis {} with size {}",
            start, end, axis, *dim,
        );

        *dim = end - start;
        self.push(new_shape, Operation::Slice { input, axis, start, end })
    }

    /// Index along a given axis.
    /// Similar to slice with a 1-sized interval except that the the resulting value doesn't have the extra axis.
    #[must_use]
    pub fn index(&mut self, input: Value, axis: usize, index: usize) -> Value {
        let sliced = self.slice(input, axis, index, index + 1);

        let mut new_shape = self[input].shape.clone();
        new_shape.dims.remove(axis);

        self.view(sliced, new_shape)
    }

    /// Index along the given axis with indices given by `index`.
    #[must_use]
    pub fn gather(&mut self, input: Value, axis: usize, indices: Value) -> Value {
        let input_shape = &self[input].shape;
        let index_size = self[indices].shape.unwrap_1();

        let mut result_shape = input_shape.clone();
        result_shape.dims[axis] = index_size;

        self.push(result_shape, Operation::Gather { input, axis, indices })
    }

    /// Concatenate values along an axis.
    #[must_use]
    pub fn concat(&mut self, inputs: Vec<Value>, axis: usize) -> Value {
        assert!(inputs.len() > 0, "Must concatenate at least one value");

        let base_shape = self[inputs[0]].shape.with_one_at(axis);

        let size_along_axis = inputs.iter().map(|&v| {
            assert_eq!(self[v].shape.with_one_at(axis), base_shape, "All concatenated values must match base shape");
            self[v].shape.dims[axis].unwrap_fixed("Size along concatenated axis")
        }).sum::<usize>();

        let mut result_shape = base_shape.clone();
        result_shape[axis] = Size::fixed(size_along_axis);

        self.push(result_shape, Operation::Concat { inputs, axis })
    }

    /// Apply 2D convolution.
    #[must_use]
    pub fn conv(&mut self, input: Value, filter: Value, padding_y: usize, padding_x: usize) -> Value {
        let [batch_size, in_c, in_h, in_w]: [Size; 4] = self[input].shape.dims.as_slice().try_into()
            .expect("Convolution input must have rank 4");
        let [out_c, in_c_check, k_h, k_w]: [Size; 4] = self[filter].shape.dims.as_slice().try_into()
            .expect("Convolution filter must have rank 4");

        // almost everything must be fixed, except for the batch size n
        let input_channels = in_c.unwrap_fixed("Conv input channels");
        let input_h = in_h.unwrap_fixed("Conv input height");
        let input_w = in_w.unwrap_fixed("Conv input width");
        let output_channels = out_c.unwrap_fixed("Conv output channels");
        let in_c_check = in_c_check.unwrap_fixed("Filter input channels");
        let kernel_h = k_h.unwrap_fixed("Conv kernel height");
        let kernel_w = k_w.unwrap_fixed("Conv kernel width");

        assert_eq!(1, kernel_h % 2, "Kernel height must be odd, got {}", kernel_h);
        assert_eq!(1, kernel_w % 2, "Kernel width must be odd, got {}", kernel_w);

        assert_eq!(input_channels, in_c_check, "Input channel mismatch");

        let output_h = input_h - kernel_h + 1 + 2 * padding_y;
        let output_w = input_w - kernel_w + 1 + 2 * padding_x;
        let output_shape = shape![batch_size, output_channels, output_h, output_w];

        let details = ConvDetails {
            batch_size,
            input_channels,
            output_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            padding_y,
            padding_x,
            output_h,
            output_w,
        };
        self.push(
            output_shape,
            Operation::Conv { input, details, filter },
        )
    }

    /// Apply a linear transformation.
    /// Input shape `[N, Ci]` and weight shape `[Co, Ci]` result in an output with shape `[N, Co]`.
    #[must_use]
    pub fn linear(&mut self, input: Value, weight: Value) -> Value {
        let input_shape = self[input].shape.unwrap_2();
        let weight_shape = self[weight].shape.unwrap_2();

        let n = input_shape[0];
        let ci = input_shape[1];
        let co = weight_shape[0];
        assert_eq!(ci, weight_shape[1]);

        // convert this linear operation into the equivalent convolution
        let input_view_shape = shape![n, ci, 1, 1];
        let input_view = self.view(input, input_view_shape);
        let weight_view_shape = shape![co, ci, 1, 1];
        let weight_view = self.view(weight, weight_view_shape);
        let output_view_shape = shape![n, co];

        let output = self.conv(input_view, weight_view, 0, 0);
        self.view(output, output_view_shape)
    }

    /// Batched matrix multiply. Inputs must have shapes `[N, p, q]`, `[N, q, r]` and the result has shape `[N, p, r]`.
    #[must_use]
    pub fn mat_mul(&mut self, left: Value, right: Value) -> Value {
        let [n0, p, q0] = self[left].shape.unwrap_3();
        let [n1, q1, r] = self[right].shape.unwrap_3();

        assert!(n0 == n1 && q0 == q1, "MatMul dimension mismatch: {:?} and {:?}", self[left].shape, self[right].shape);

        let result_shape = shape![n0, p, r];
        self.push(result_shape, Operation::MatMul { left, right })
    }

    /// Elementwise clamp.
    #[must_use]
    pub fn clamp(&mut self, input: Value, min: f32, max: f32) -> Value {
        if min == f32::NEG_INFINITY && max == f32::INFINITY {
            return input;
        }

        self.push(self[input].shape.clone(), Operation::Clamp { input, min, max })
    }

    /// Elementwise relu.
    #[must_use]
    pub fn relu(&mut self, input: Value) -> Value {
        self.clamp(input, 0.0, f32::INFINITY)
    }

    /// Add two values together elementwise.
    /// They must have the same rank, and the right shape is broadcasted to the left shape.
    #[must_use]
    pub fn add(&mut self, left: Value, right: Value) -> Value {
        let output_shape = self.check_broadcast(left, right);

        if self.is_all_zero(right) {
            return left;
        }

        self.push(output_shape, Operation::Add { left, right, subtract: false })
    }

    /// Subtract two values elementwise.
    /// They must have the same rank, and the right shape is broadcasted to the left shape.
    #[must_use]
    pub fn sub(&mut self, left: Value, right: Value) -> Value {
        let output_shape = self.check_broadcast(left, right);

        if self.is_all_zero(right) {
            return left;
        }

        self.push(output_shape, Operation::Add { left, right, subtract: true })
    }

    /// Multiple two values elementwise.
    /// They must have the same rank, and the right shape is broadcasted to the left shape.
    #[must_use]
    pub fn mul(&mut self, left: Value, right: Value) -> Value {
        let output_shape = self.check_broadcast(left, right);

        if self.is_all_one(right) {
            return left;
        }

        self.push(output_shape, Operation::Mul { left, right })
    }

    /// Register an existing value as an output
    pub fn output(&mut self, value: Value) {
        self.outputs.push(value);
    }

    /// Register multiple values as output at once, in order.
    pub fn output_all(&mut self, values: &[Value]) {
        for &value in values {
            self.output(value)
        }
    }
}

impl Debug for Graph {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph")
            .field("inputs", &self.inputs().iter().map(|&v| &self[v].shape).collect_vec())
            .field("outputs", &self.outputs().iter().map(|&v| &self[v].shape).collect_vec())
            .finish_non_exhaustive()
    }
}

impl Display for Graph {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let Graph { values, inputs, outputs } = self;

        writeln!(f, "Graph {{")?;

        writeln!(f, "  values: [")?;
        for (i, info) in values.iter().enumerate() {
            writeln!(f, "    {:?} = {:?},", Value(i), info)?;
        }
        writeln!(f, "  ],")?;

        writeln!(f, "  inputs: {:?},", inputs)?;
        writeln!(f, "  outputs: {:?},", outputs)?;

        writeln!(f, "}}")?;

        Ok(())
    }
}

impl Debug for ConstantData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.0.len() <= 16 {
            write!(f, "{:?}", self.0)
        } else {
            write!(f, "[..; {}]", self.0.len())
        }
    }
}

impl Deref for ConstantData {
    type Target = Vec<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}