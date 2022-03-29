use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Deref, Index};

use decorum::cmp::FloatEq;
use itertools::{zip_eq, Itertools};
use rand::{thread_rng, Rng};

use crate::shape;
use crate::shape::{Shape, Size};

#[derive(Clone)]
pub struct Graph {
    check: u32,
    values: Vec<ValueInfo>,
    inputs: Vec<Value>,
    outputs: Vec<Value>,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Value {
    index: usize,
    check: u32,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ValueInfo {
    pub shape: Shape,
    pub operation: Operation,
}

/// Wrapper type that prevents the Debug output from getting too large.
#[derive(Clone)]
pub struct ConstantData(Vec<f32>);

/// The core set of graph operations.
/// Some attempt was made to keep operations orthogonal but flexible, so they can be composed easily.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Operation {
    /// A runtime-variable input.
    Input { index: usize },
    /// A constant build into the network.
    Constant { data: ConstantData },

    //TODO maybe fuse a bunch of these operations into a single "Restride" operation?
    /// View a value as a different shape.
    View { input: Value },
    /// Change the order of axis in the shape.
    Permute { input: Value, permutation: Vec<usize> },
    /// Slice along the given `axis` with range `start..end`.
    Slice {
        input: Value,
        axis: usize,
        range: SliceRange,
    },
    /// Flip the given axis.
    Flip { input: Value, axis: usize },

    /// Gather values from `input` at the indices in `index` on the given axis.
    Gather { input: Value, axis: usize, indices: Value },

    /// Concatenate values along an axis.
    Concat { inputs: Vec<Value>, axis: usize },

    /// The standard convolution operator.
    Conv {
        input: Value,
        filter: Value,
        details: ConvDetails,
    },
    /// Batched matrix multiply.
    MatMul { left: Value, right: Value },

    /// Elementwise operation between two operands, with broadcasting on the right.
    Element { left: Value, right: Value, op: ElementOp },
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct SliceRange {
    pub start: usize,
    pub end: usize,
    pub step: usize,
}

/// An elementwise operation.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ElementOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

impl Operation {
    pub fn inputs(&self) -> Vec<Value> {
        match self {
            Operation::Input { index: _ } => vec![],
            Operation::Constant { data: _ } => vec![],
            &Operation::View { input } => vec![input],
            &Operation::Permute { input, permutation: _ } => vec![input],
            &Operation::Slice {
                input,
                axis: _,
                range: _,
            } => vec![input],
            &Operation::Flip { input, axis: _ } => vec![input],
            &Operation::Gather {
                input,
                axis: _,
                indices,
            } => vec![input, indices],
            Operation::Concat { inputs, axis: _ } => inputs.clone(),
            &Operation::Conv {
                input,
                filter,
                details: _,
            } => vec![input, filter],
            &Operation::MatMul { left, right } => vec![left, right],
            &Operation::Element { left, right, op: _ } => vec![left, right],
        }
    }

    pub(crate) fn clone_map_inputs(&self, mut f: impl FnMut(Value) -> Value) -> Operation {
        match self {
            &Operation::Input { index } => Operation::Input { index },
            Operation::Constant { data } => Operation::Constant { data: data.clone() },
            &Operation::View { input } => Operation::View { input: f(input) },
            &Operation::Permute { input, ref permutation } => Operation::Permute {
                input: f(input),
                permutation: permutation.clone(),
            },
            &Operation::Slice { input, axis, range } => Operation::Slice {
                input: f(input),
                axis,
                range,
            },
            &Operation::Flip { input, axis } => Operation::Flip { input: f(input), axis },
            &Operation::Gather { input, axis, indices } => Operation::Gather {
                input: f(input),
                axis,
                indices: f(indices),
            },
            &Operation::Concat { ref inputs, axis } => Operation::Concat {
                inputs: inputs.iter().copied().map(f).collect(),
                axis,
            },
            &Operation::Conv {
                input,
                filter,
                details: conv_shape,
            } => Operation::Conv {
                input: f(input),
                filter: f(filter),
                details: conv_shape,
            },
            &Operation::MatMul { left, right } => Operation::MatMul {
                left: f(left),
                right: f(right),
            },
            &Operation::Element { left, right, op } => Operation::Element {
                left: f(left),
                right: f(right),
                op,
            },
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
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

    pub fn kernel_shape(&self) -> [usize; 4] {
        [self.output_channels, self.input_channels, self.kernel_h, self.kernel_w]
    }
}

impl Index<Value> for Graph {
    type Output = ValueInfo;

    fn index(&self, value: Value) -> &Self::Output {
        self.check_contains(value);
        &self.values[value.index]
    }
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            check: thread_rng().gen(),
            values: vec![],
            inputs: vec![],
            outputs: vec![],
        }
    }

    fn check_contains(&self, value: Value) {
        assert_eq!(
            value.check, self.check,
            "Value {:?} does not belong to this graph",
            value
        );
        assert!(value.index < self.values.len());
    }

    fn broadcast(&mut self, left: Value, right: Value) -> Value {
        let left_shape = &self[left].shape;
        let right_shape = &self[right].shape;

        if right_shape.rank() == 0 {
            let new_right_shape = Shape::ones(left_shape.rank());
            self.view(right, new_right_shape)
        } else {
            assert_eq!(
                left_shape.rank(),
                right_shape.rank(),
                "Both inputs must have the same rank (or right must have rank 0), got {:?} and {:?}",
                left_shape,
                right_shape
            );

            for (&l, &r) in zip_eq(&left_shape.dims, &right_shape.dims) {
                assert!(
                    l == r || r == Size::ONE,
                    "Cannot broadcast shape {:?} to {:?}",
                    right_shape,
                    left_shape
                );
            }

            right
        }
    }

    /// Iterate over the values in this graph, in topological order,
    /// which means that nodes will only be visited after all of their inputs have been visited.
    pub fn values(&self) -> impl Iterator<Item = Value> {
        let check = self.check;
        (0..self.values.len()).map(move |index| Value { index, check })
    }

    pub fn inputs(&self) -> &[Value] {
        &self.inputs
    }

    pub fn input_shapes(&self) -> Vec<Shape> {
        self.inputs().iter().map(|&v| self[v].shape.clone()).collect()
    }

    pub fn outputs(&self) -> &[Value] {
        &self.outputs
    }

    pub fn output_shapes(&self) -> Vec<Shape> {
        self.outputs().iter().map(|&v| self[v].shape.clone()).collect()
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

    pub fn is_const_filled_with(&self, value: Value, f: f32) -> bool {
        self.as_const(value).map_or(false, |x| x.iter().all(|&x| x == f))
    }

    #[must_use]
    pub(crate) fn push(&mut self, shape: Shape, operation: Operation) -> Value {
        let info = ValueInfo { shape, operation };

        let index = match self.values.iter().position(|cand| cand == &info) {
            Some(index) => {
                // found duplicate, reuse existing value
                index
            }
            None => {
                for input in info.operation.inputs() {
                    self.check_contains(input);
                }

                let index = self.values.len();
                self.values.push(info);
                index
            }
        };

        Value {
            index,
            check: self.check,
        }
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
        assert_eq!(
            expected_len,
            data.len() as usize,
            "Shape {:?} and data size {} mismatch",
            shape,
            data.len()
        );

        self.push(
            shape,
            Operation::Constant {
                data: ConstantData(data),
            },
        )
    }

    /// View an existing value as a new shape.
    #[must_use]
    pub fn view(&mut self, input: Value, new_shape: Shape) -> Value {
        let old_shape = &self[input].shape;
        if &new_shape == old_shape {
            return input;
        }

        assert_eq!(
            old_shape.size(),
            new_shape.size(),
            "New shape {:?} must have the same size as old shape {:?}",
            new_shape,
            old_shape,
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
            "Input rank {} to low for start axis {}",
            old_shape.rank(),
            start_axis
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

        assert_eq!(
            permutation.len(),
            input_shape.rank(),
            "Permutation rank must match input shape, got {:?} and {:?}",
            permutation,
            input_shape
        );
        assert!(
            permutation.iter().all_unique(),
            "Permutation cannot contain repeated axis, got {:?}",
            permutation
        );
        assert!(
            permutation.iter().all(|&i| i < input_shape.rank()),
            "Permutation axis out of bounds, got {:?}",
            permutation
        );

        let result_dims = permutation.iter().map(|&i| input_shape[i]).collect_vec();
        let result_shape = Shape::new(result_dims);

        self.push(result_shape, Operation::Permute { input, permutation })
    }

    /// Slice a value along an axis. See [slice_range] for a more general version.
    #[must_use]
    pub fn slice(&mut self, input: Value, axis: usize, range: SliceRange) -> Value {
        range.assert_valid();

        let old_shape = &self[input].shape;
        assert!(
            axis < old_shape.rank(),
            "Slice axis {} out of bounds for {:?}",
            axis,
            old_shape,
        );

        let old_size = old_shape.dims[axis].unwrap_fixed("Slice axis length");
        let new_size = (range.end - range.start) / range.step;

        // skip trivial slice
        if range == SliceRange::new(0, old_size, 1) {
            return input;
        }

        let new_shape = old_shape.replace(axis, Size::fixed(new_size));
        self.push(new_shape, Operation::Slice { input, axis, range })
    }

    /// Index along a given axis.
    /// Similar to slice with a 1-sized interval except that the the resulting value doesn't have the extra axis.
    #[must_use]
    pub fn index(&mut self, input: Value, axis: usize, index: usize) -> Value {
        let sliced = self.slice(input, axis, SliceRange::single(index));

        let mut new_shape = self[input].shape.clone();
        new_shape.dims.remove(axis);

        self.view(sliced, new_shape)
    }

    /// Flip the given `axis`.
    pub fn flip(&mut self, input: Value, axis: usize) -> Value {
        let shape = self[input].shape.clone();
        assert!(axis <= shape.rank(), "Axis {} out of bounds for {:?}", axis, shape);

        self.push(shape, Operation::Flip { input, axis })
    }

    /// Repeat `input` along a given `axis`, `count` times.
    pub fn repeat(&mut self, input: Value, axis: usize, count: usize) -> Value {
        //TODO introduce separate (optimized) operation for this?
        //TODO special-case length-0 axis to some kind of restride operation
        let base_shape = self[input].shape.replace(axis, Size::ZERO);
        self.concat(vec![input; count], axis, Some(base_shape))
    }

    /// Index along the given `axis` with indices given by `index`.
    #[must_use]
    pub fn gather(&mut self, input: Value, axis: usize, indices: Value) -> Value {
        let input_shape = &self[input].shape;
        let index_size = self[indices].shape.unwrap_1();

        let mut result_shape = input_shape.clone();
        result_shape.dims[axis] = index_size;

        self.push(result_shape, Operation::Gather { input, axis, indices })
    }

    /// Concatenate `inputs` along `axis`.
    /// `base_shape` can be provided to allow the result shape to be inferred in case `inputs` is empty.
    #[must_use]
    pub fn concat(&mut self, inputs: Vec<Value>, axis: usize, base_shape: Option<Shape>) -> Value {
        let base_shape = base_shape.unwrap_or_else(|| {
            assert!(
                !inputs.is_empty(),
                "Cannot infer concatenation shape without any values"
            );
            self[inputs[0]].shape.replace(axis, Size::ZERO)
        });

        let size_along_axis = inputs
            .iter()
            .map(|&v| {
                assert_eq!(
                    self[v].shape.replace(axis, Size::ZERO),
                    base_shape,
                    "All concatenated values must match base shape on non-concatenated axes"
                );
                self[v].shape.dims[axis].unwrap_fixed("Size along concatenated axis")
            })
            .sum::<usize>();

        let mut result_shape = base_shape;
        result_shape[axis] = Size::fixed(size_along_axis);

        self.push(result_shape, Operation::Concat { inputs, axis })
    }

    /// Apply 2D convolution.
    #[must_use]
    pub fn conv(&mut self, input: Value, filter: Value, padding_y: usize, padding_x: usize) -> Value {
        let [batch_size, in_c, in_h, in_w]: [Size; 4] = self[input]
            .shape
            .dims
            .as_slice()
            .try_into()
            .expect("Convolution input must have rank 4");
        let [out_c, in_c_check, k_h, k_w]: [Size; 4] = self[filter]
            .shape
            .dims
            .as_slice()
            .try_into()
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
        self.push(output_shape, Operation::Conv { input, details, filter })
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

        assert!(
            n0 == n1 && q0 == q1,
            "MatMul dimension mismatch: {:?} and {:?}",
            self[left].shape,
            self[right].shape
        );

        let result_shape = shape![n0, p, r];
        self.push(result_shape, Operation::MatMul { left, right })
    }

    /// Elementwise relu..
    #[must_use]
    pub fn relu(&mut self, input: Value) -> Value {
        self.clamp(input, 0.0, f32::INFINITY)
    }

    /// Elementwise clamp.
    #[must_use]
    pub fn clamp(&mut self, input: Value, min: f32, max: f32) -> Value {
        // careful, min/max are intentionally flipped to yield MAX(MIN(x, max), min)
        let right_shape = Shape::ones(self[input].shape.rank());

        let mut curr = input;

        // these checks are kind of tedious but it prevents the value allocations if they're not necessary
        if max != f32::INFINITY {
            let max_value = self.constant(right_shape.clone(), vec![max]);
            curr = self.ele(ElementOp::Min, curr, max_value);
        }

        if min != f32::INFINITY {
            let min_value = self.constant(right_shape, vec![min]);
            curr = self.ele(ElementOp::Max, curr, min_value);
        }

        curr
    }

    #[must_use]
    pub fn add(&mut self, left: Value, right: Value) -> Value {
        self.ele(ElementOp::Add, left, right)
    }

    #[must_use]
    pub fn sub(&mut self, left: Value, right: Value) -> Value {
        self.ele(ElementOp::Sub, left, right)
    }

    #[must_use]
    pub fn mul(&mut self, left: Value, right: Value) -> Value {
        self.ele(ElementOp::Mul, left, right)
    }

    /// Compute an elementwise operation between two values.
    /// They must have the same rank (or right must have rank 0), the right shape is broadcasted to the left shape.
    #[must_use]
    pub fn ele(&mut self, op: ElementOp, left: Value, right: Value) -> Value {
        let right = self.broadcast(left, right);

        let skip = match op {
            ElementOp::Sub | ElementOp::Add => self.is_const_filled_with(right, 0.0),
            ElementOp::Mul | ElementOp::Div => self.is_const_filled_with(right, 1.0),
            ElementOp::Min => self.is_const_filled_with(right, f32::INFINITY),
            ElementOp::Max => self.is_const_filled_with(right, f32::NEG_INFINITY),
        };
        if skip {
            return left;
        }

        let result_shape = self[left].shape.clone();
        self.push(result_shape, Operation::Element { left, right, op })
    }

    /// Computes the operations described by `graph` on the given inputs.
    #[must_use]
    pub fn call(&mut self, graph: &Graph, inputs: &[Value]) -> Vec<Value> {
        let mut map = HashMap::new();

        // check inputs
        assert_eq!(inputs.len(), graph.inputs.len(), "Wrong number of inputs");
        for (&input, &graph_input) in zip_eq(inputs, &graph.inputs) {
            assert_eq!(self[input].shape, graph[graph_input].shape, "Wrong input shape");
        }

        // map operations
        for graph_value in graph.values() {
            let graph_info = &graph[graph_value];

            let shape = graph_info.shape.clone();
            let graph_operation = &graph_info.operation;

            let value = if let &Operation::Input { index } = graph_operation {
                inputs[index]
            } else {
                let operation = graph_info.operation.clone_map_inputs(|p| *map.get(&p).unwrap());
                self.push(shape, operation)
            };

            map.insert(graph_value, value);
        }

        // map outputs
        graph
            .outputs()
            .iter()
            .map(|graph_value| *map.get(graph_value).unwrap())
            .collect_vec()
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
        let Graph {
            check,
            values,
            inputs,
            outputs,
        } = self;

        writeln!(f, "Graph {{")?;

        writeln!(f, "  check: {},", self.check)?;
        writeln!(f, "  inputs: {:?},", inputs)?;
        writeln!(f, "  outputs: {:?},", outputs)?;

        writeln!(f, "  values: [")?;
        for (i, info) in values.iter().enumerate() {
            writeln!(
                f,
                "    {:?} = {:?},",
                Value {
                    index: i,
                    check: *check
                },
                info
            )?;
        }
        writeln!(f, "  ],")?;

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

impl PartialEq for ConstantData {
    fn eq(&self, other: &Self) -> bool {
        self.0.float_eq(&other.0)
    }
}

impl Eq for ConstantData {}

impl Deref for ConstantData {
    type Target = Vec<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Value {
    pub fn index(self) -> usize {
        self.index
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let Value { index, check } = self;
        if f.alternate() {
            write!(f, "Value {{ index: {}, check: {} }}", index, check)
        } else {
            write!(f, "Value({})", index)
        }
    }
}

impl SliceRange {
    pub fn new(start: usize, end: usize, step: usize) -> Self {
        let result = Self { start, end, step };
        result.assert_valid();
        result
    }

    pub fn simple(start: usize, end: usize) -> Self {
        Self::new(start, end, 1)
    }

    pub fn single(index: usize) -> Self {
        Self::new(index, index + 1, 1)
    }

    pub fn empty() -> Self {
        Self::new(0, 0, 1)
    }

    pub fn assert_valid(self) {
        assert!(
            self.end >= self.start,
            "Invalid range {:?}: bounds cannot be decreasing",
            self,
        );

        assert_ne!(self.step, 0, "Invalid range {:?}: step cannot be 0", self);

        assert_eq!(
            (self.end - self.start) % self.step,
            0,
            "Invalid range {:?}: bounds must differ by a multiple of step",
            self
        );
    }
}
