use std::cmp::max;
use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Deref, Index};

use decorum::cmp::FloatEq;
use decorum::Total;
use itertools::{zip_eq, Itertools};
use rand::random;

use crate::cpu::{run_cpu_const_operation, OperationError, Tensor};
use crate::shape;
use crate::shape::{Shape, Size};

/// The core graph datastructure.
///
/// This is a Directed Acyclic Graph (DAG) with values and their creating operations as nodes,
/// and input operands as edges. The data structure is append-only, values cannot be removed
/// and so will never become invalid.
///
/// This type implements `Index<Value>` trait, so you can use `graph[value]` to get information about the given value.
///
/// ```
/// # use kn_graph::graph::*;
/// # use kn_graph::shape;
/// # use kn_graph::shape::*;
/// // create a new graph
/// let mut graph = Graph::new();
///
/// // define the inputs
/// let x = graph.input(shape![Size::BATCH, 4, 8, 8]);
///
/// // define constants
/// let w_data = vec![0.5; 4 * 4 * 3 * 3];
/// let w = graph.constant(shape![4, 4, 3, 3], w_data);
/// let b_data = vec![0.5; 4];
/// let b = graph.constant(shape![4, 1, 1], b_data);
///
/// // build operation graph
/// let y0 = graph.conv(x, w, 1, 1, 1, 1);
/// let y = graph.add(y0, b);
///
/// graph.output(y);
///
/// println!("{}", graph);
/// ```
/// Results in the following output:
/// ```text
/// Graph {
///   check: 1504812640,
///   input_shapes: [Shape(B x 4 x 8 x 8)],
///   output_shapes: [Shape(B x 4 x 8 x 8)],
///   inputs: [Value(0)],
///   outputs: [Value(6)],
///   values: [
///     Value(0) = ValueInfo { shape: Shape(B x 4 x 8 x 8), operation: Input { index: 0 }, debug_id: "", non_output_uses: 1 },
///     Value(1) = ValueInfo { shape: Shape(4 x 4 x 3 x 3), operation: Constant { data: [..; 144] }, debug_id: "", non_output_uses: 1 },
///     Value(2) = ValueInfo { shape: Shape(4 x 1 x 1), operation: Constant { data: [0.5, 0.5, 0.5, 0.5] }, debug_id: "", non_output_uses: 1 },
///     Value(3) = ValueInfo { shape: Shape(B x 4 x 8 x 8), operation: Conv { input: Value(0), filter: Value(1), details: ConvDetails { batch_size: Size(B), input_channels: 4, output_channels: 4, input_h: 8, input_w: 8, kernel_h: 3, kernel_w: 3, stride_y: 1, stride_x: 1, padding_y: 1, padding_x: 1, output_h: 8, output_w: 8 } }, debug_id: "", non_output_uses: 1 },
///     Value(4) = ValueInfo { shape: Shape(1 x 4 x 1 x 1), operation: View { input: Value(2) }, debug_id: "", non_output_uses: 1 },
///     Value(5) = ValueInfo { shape: Shape(B x 4 x 8 x 8), operation: Broadcast { input: Value(4) }, debug_id: "", non_output_uses: 1 },
///     Value(6) = ValueInfo { shape: Shape(B x 4 x 8 x 8), operation: Binary { left: Value(3), right: Value(5), op: Add }, debug_id: "", non_output_uses: 0 },
///   ],
/// }
/// ```
#[derive(Clone)]
pub struct Graph {
    check: u32,
    values: Vec<ValueInfo>,
    new_values: Vec<Value>,
    inputs: Vec<Value>,
    outputs: Vec<Value>,
}

/// A value in a [Graph].
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Value {
    index: usize,
    check: u32,
}

/// Information about a [Value], most importantly its shape and creating operation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ValueInfo {
    pub shape: Shape,
    pub operation: Operation,
    pub debug_id: String,
    non_output_uses: usize,
}

/// Wrapper type that prevents the Debug output from getting too large.
#[derive(Clone)]
pub struct ConstantData(pub Vec<f32>);

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
    /// Repeat along all axes with size 1 that don't match the output shape.
    Broadcast { input: Value },
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

    /// Gather values from `input` at the indices in `index` on the given `axis`.
    /// `indices` is a rank-1 tensor.
    Gather { input: Value, axis: usize, indices: Value },

    /// Concatenate values along an axis.
    Concat { inputs: Vec<Value>, axis: usize },

    /// 2D convolution.
    Conv {
        input: Value,
        filter: Value,
        details: ConvDetails,
    },
    /// (Batched) Matrix multiply.
    /// If left has shape `[b, p, q]` and right has shape `[b, q, r]` the result has shape `[b, p, r]`.
    MatMul { left: Value, right: Value },

    /// Elementwise unary operation.
    Unary { input: Value, op: UnaryOp },
    /// Elementwise binary operation. Both operands must have the same shape.
    Binary { left: Value, right: Value, op: BinaryOp },

    /// Softmax along `axis`.
    Softmax { input: Value, axis: usize },
    /// Layernorm along `axis`.
    Layernorm { input: Value, axis: usize, eps: Total<f32> },

    /// Reduce along the given `axes` using `op`. The `axes` are removed from the shape.
    Reduce {
        input: Value,
        axes: Vec<usize>,
        op: ReduceOp,
    },
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct SliceRange {
    pub start: usize,
    pub end: usize,
    pub step: usize,
}

// TODO consider removing the compound operations (sigmoid, mish)
//   alternatively check if either the CPU or CUDA implementations are faster/more accurate
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum UnaryOp {
    Abs,
    Neg,
    Sin,
    Cos,
    Exp,
    Log,
    Sqrt,
    Sigmoid,
    Tanh,
    Erf,
    Mish,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Pow,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Prod,
    Max,
    Min,
}

impl Operation {
    pub fn inputs(&self) -> Vec<Value> {
        match self {
            Operation::Input { index: _ } => vec![],
            Operation::Constant { data: _ } => vec![],
            &Operation::View { input } => vec![input],
            &Operation::Broadcast { input } => vec![input],
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
            &Operation::Unary { input, op: _ } => vec![input],
            &Operation::Binary { left, right, op: _ } => vec![left, right],
            &Operation::Softmax { input, axis: _ } => vec![input],
            &Operation::Layernorm { input, axis: _, eps: _ } => vec![input],
            &Operation::Reduce { input, axes: _, op: _ } => vec![input],
        }
    }

    pub(crate) fn clone_map_inputs(&self, mut f: impl FnMut(Value) -> Value) -> Operation {
        match self {
            &Operation::Input { index } => Operation::Input { index },
            &Operation::Constant { ref data } => Operation::Constant { data: data.clone() },
            &Operation::View { input } => Operation::View { input: f(input) },
            &Operation::Broadcast { input } => Operation::Broadcast { input: f(input) },
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
            &Operation::Unary { input, op } => Operation::Unary { input: f(input), op },
            &Operation::Binary { left, right, op } => Operation::Binary {
                left: f(left),
                right: f(right),
                op,
            },
            &Operation::Softmax { input, axis } => Operation::Softmax { input: f(input), axis },
            &Operation::Layernorm { input, axis, eps } => Operation::Layernorm {
                input: f(input),
                axis,
                eps,
            },
            &Operation::Reduce { input, ref axes, op } => Operation::Reduce {
                input: f(input),
                axes: axes.clone(),
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
    pub stride_y: usize,
    pub stride_x: usize,
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

    pub fn has_stride(&self) -> bool {
        self.stride_y != 1 || self.stride_x != 1
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
            check: random(),
            values: vec![],
            new_values: vec![],
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

    pub fn is_hidden(&self, value: Value) -> bool {
        self.check_contains(value);
        !self.inputs.contains(&value) && !self.outputs.contains(&value)
    }

    pub fn is_hidden_with_uses(&self, value: Value, users: usize) -> bool {
        self.is_hidden(value) && self[value].non_output_uses == users
    }

    pub fn is_const(&self, value: Value) -> bool {
        let operation = &self[value].operation;
        match *operation {
            Operation::Input { .. } => false,
            Operation::Constant { .. } => true,
            _ => operation.inputs().into_iter().all(|input| self.is_const(input)),
        }
    }

    /// Try to evaluate `value` as a constant.
    pub fn as_const(&self, value: Value) -> Option<Tensor> {
        run_cpu_const_operation(&self[value], |x| self.as_const(x).ok_or(OperationError::MissingOperand)).ok()
    }

    /// Returns whether `value` is effectively a constant with every element equal to `f`.
    pub fn is_const_filled_with(&self, value: Value, f: f32) -> bool {
        self.as_single_const(value).map_or(false, |g| f.float_eq(&g))
    }

    /// Returns `Some(f)` if `value` is effectively a constant with every element equal to `f`.
    pub fn as_single_const(&self, value: Value) -> Option<f32> {
        match &self[value].operation {
            Operation::Input { .. } => None,
            Operation::Constant { data } => {
                let f = *data.first()?;
                data.iter().all(|&x| f.float_eq(&x)).then(|| f)
            }
            &Operation::View { input } => self.as_single_const(input),
            &Operation::Broadcast { input } => self.as_single_const(input),
            &Operation::Permute { input, permutation: _ } => self.as_single_const(input),
            &Operation::Slice {
                input,
                axis: _,
                range: _,
            } => self.as_single_const(input),
            &Operation::Flip { input, axis: _ } => self.as_single_const(input),
            &Operation::Gather {
                input,
                axis: _,
                indices: _,
            } => self.as_single_const(input),
            &Operation::Concat { ref inputs, axis: _ } => {
                let f = self.as_single_const(*inputs.first()?)?;
                inputs.iter().all(|&x| self.is_const_filled_with(x, f)).then(|| f)
            }
            Operation::Conv { .. }
            | Operation::MatMul { .. }
            | Operation::Unary { .. }
            | Operation::Binary { .. }
            | Operation::Softmax { .. }
            | Operation::Layernorm { .. }
            | Operation::Reduce { .. } => None,
        }
    }

    /// Return all newly crated values since the last call to `take_new_values`.
    pub fn take_new_values(&mut self) -> Vec<Value> {
        std::mem::take(&mut self.new_values)
    }

    #[must_use]
    pub(crate) fn push(&mut self, shape: Shape, operation: Operation) -> Value {
        let info = ValueInfo {
            shape,
            operation,
            non_output_uses: 0,
            debug_id: String::new(),
        };

        let check = self.check;

        match self.values.iter().position(|cand| cand == &info) {
            Some(index) => {
                // found duplicate, reuse existing value
                Value { index, check }
            }
            None => {
                // no duplicate found, create new value
                for input in info.operation.inputs() {
                    self.check_contains(input);
                    self.values[input.index].non_output_uses += 1;
                }

                let index = self.values.len();
                let value = Value { index, check };

                self.values.push(info);
                self.new_values.push(value);

                value
            }
        }
    }

    /// Equivalent to `self[value].debug_id = id`,
    /// but that would not work since there is intentionally no implementation of `IndexMut` for `Graph`.
    pub fn set_debug_id(&mut self, value: Value, id: String) {
        self.check_contains(value);
        self.values[value.index].debug_id = id;
    }

    /// Declare a new input value.
    #[must_use]
    pub fn input(&mut self, shape: Shape) -> Value {
        let index = self.inputs.len();
        let value = self.push(shape, Operation::Input { index });
        self.inputs.push(value);
        value
    }

    #[must_use]
    pub fn scalar(&mut self, value: f32) -> Value {
        self.constant(Shape::SCALAR, vec![value])
    }

    /// Declare a new constant.
    #[must_use]
    pub fn constant(&mut self, shape: Shape, data: Vec<f32>) -> Value {
        let expected_len = shape.unwrap_fixed("Constant shape must be fixed").size();
        assert_eq!(
            expected_len,
            data.len(),
            "{:?} has size {}, but got data with size {}",
            shape,
            expected_len,
            data.len()
        );

        self.push(
            shape,
            Operation::Constant {
                data: ConstantData(data),
            },
        )
    }

    #[must_use]
    pub fn constant_tensor(&mut self, tensor: Tensor) -> Value {
        let shape = Shape::fixed(tensor.shape());
        let data = tensor.iter().copied().collect();
        self.constant(shape, data)
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

        // only keep the last view operation
        let inner_input = if let &Operation::View { input: inner_input } = &self[input].operation {
            inner_input
        } else {
            input
        };

        self.push(new_shape, Operation::View { input: inner_input })
    }

    /// Broadcast the `input` towards `new_shape`.
    /// Additional unit axes are are inserted at the front and unit axes are repeated as necessary.
    #[must_use]
    pub fn broadcast(&mut self, input: Value, new_shape: Shape) -> Value {
        let input_shape = &self[input].shape.clone();

        assert!(
            input_shape.rank() <= new_shape.rank(),
            "Cannot broadcast to a lower rank shape (from {:?} to {:?})",
            input_shape,
            new_shape
        );

        // pad with 1 axes
        let view_shape = Shape::ones(new_shape.rank() - input_shape.rank()).concat(&input_shape);
        let curr = self.view(input, view_shape.clone());

        // check that broadcasting is valid)
        for (&v, &n) in zip_eq(&view_shape.dims, &new_shape.dims) {
            assert!(
                v == n || v == Size::ONE,
                "Cannot broadcast from {:?} to {:?} because of axis ({}, {})",
                input_shape,
                new_shape,
                v,
                n
            );
        }

        // don't need to actually broadcast
        if view_shape == new_shape {
            return curr;
        }

        // do the actual broadcast
        self.push(new_shape, Operation::Broadcast { input: curr })
    }

    pub fn repeat_unary(&mut self, input: Value, axis: usize, count: Size) -> Value {
        let input_shape = &self[input].shape;
        assert_eq!(
            input_shape[axis],
            Size::ONE,
            "Input shape {} does not have dim 1 for axis {}",
            input_shape,
            axis
        );

        // TODO fuse consecutive broadcast operations, maybe even view/broadcast/view if the axes are independent
        // skip broadcast operation
        if count == Size::ONE {
            return input;
        }

        self.push(input_shape.replace(axis, shape![count]), Operation::Broadcast { input })
    }

    /// View a value with a flattened shape.
    /// All axis starting from `start_axis` inclusive are flattened into a single axis.
    #[must_use]
    pub fn flatten(&mut self, input: Value, start_axis: usize) -> Value {
        let old_shape = &self[input].shape;
        assert!(
            start_axis <= old_shape.rank(),
            "Flatten start axis {} out of bounds for {}",
            start_axis,
            old_shape,
        );

        let kept_dims = &old_shape.dims[..start_axis];
        let flat_size = old_shape.dims[start_axis..].iter().copied().product();
        let new_shape = Shape::new([kept_dims, &[flat_size]].concat());

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

        // fuse consecutive permute operations
        let (inner_input, full_permutation) = if let &Operation::Permute {
            input: inner_input,
            permutation: ref inner_permutation,
        } = &self[input].operation
        {
            let combined = permutation.iter().map(|&i| inner_permutation[i]).collect();
            (inner_input, combined)
        } else {
            (input, permutation)
        };

        let inner_input_shape = &self[inner_input].shape;
        let result_dims = full_permutation.iter().map(|&i| inner_input_shape[i]).collect_vec();
        let result_shape = Shape::new(result_dims);

        self.push(
            result_shape,
            Operation::Permute {
                input: inner_input,
                permutation: full_permutation,
            },
        )
    }

    /// Slice a value along an axis.
    #[must_use]
    pub fn slice(&mut self, input: Value, axis: usize, range: SliceRange) -> Value {
        let old_shape = &self[input].shape;
        old_shape.assert_has_axis(axis);

        let old_size = old_shape.dims[axis].unwrap_fixed("Slice axis length");
        range.assert_in_bounds(old_size);
        let new_size = (range.end - range.start) / range.step;

        // skip trivial slice
        if range == SliceRange::new(0, old_size, 1) {
            return input;
        }

        let new_shape = old_shape.replace(axis, shape![new_size]);
        self.push(new_shape, Operation::Slice { input, axis, range })
    }

    /// Index along a given axis.
    /// Similar to slice with a 1-sized interval except that the the resulting value doesn't have the extra axis.
    #[must_use]
    pub fn index(&mut self, input: Value, axis: usize, index: usize) -> Value {
        let new_shape = self[input].shape.replace(axis, shape![]);
        let sliced = self.slice(input, axis, SliceRange::single(index));
        self.view(sliced, new_shape)
    }

    /// Flip the given `axis`.
    pub fn flip(&mut self, input: Value, axis: usize) -> Value {
        let shape = self[input].shape.clone();
        shape.assert_has_axis(axis);

        self.push(shape, Operation::Flip { input, axis })
    }

    /// Repeat `input` along a given `axis`, `count` times.
    /// This starts by emitting the entire tensor before repeating elements,
    /// similar to `torch.repeat` or `numpy.tile`.
    pub fn repeat(&mut self, input: Value, axis: usize, count: Size) -> Value {
        self.repeat_impl(input, axis, count, false)
    }

    /// Repeat elements of `input` along a given `axis`, `count` times.
    /// This starts by repeat each element before going to the next one,
    /// similar to `torch.repeat_interleave` or `numpy.repeat`.
    pub fn repeat_interleave(&mut self, input: Value, axis: usize, count: Size) -> Value {
        self.repeat_impl(input, axis, count, true)
    }

    fn repeat_impl(&mut self, input: Value, axis: usize, count: Size, inner: bool) -> Value {
        let input_shape = self[input].shape.clone();
        input_shape.assert_has_axis(axis);

        // do simpler repeat operation instead
        if input_shape[axis] == Size::ONE {
            return self.repeat_unary(input, axis, count);
        }

        let new_size = input_shape[axis] * count;
        let dummy_axis = if inner { axis + 1 } else { axis };

        // insert dummy axis, repeat dummy axis, flatten into main axis
        let extra = self.view(input, input_shape.insert(dummy_axis, Size::ONE));
        let broad = self.repeat_unary(extra, dummy_axis, count);
        let result = self.view(broad, input_shape.replace(axis, shape![new_size]));

        result
    }

    /// Index `input` along the given `axis` with indices given by `indices`.
    ///
    /// The `output` shape is the `input` shape with `axis` replaced by the shape of `indices`.
    #[must_use]
    pub fn gather(&mut self, input: Value, axis: usize, indices: Value) -> Value {
        let input_shape = &self[input].shape;
        let indices_shape = &self[indices].shape;

        input_shape.assert_has_axis(axis);

        let result_shape = input_shape.replace(axis, indices_shape.clone());
        let result_shape_flat = input_shape.replace(axis, shape![indices_shape.size()]);

        // we support arbitrary rank indices here, but the actual operation does not
        let flat_indices = self.flatten(indices, 0);
        let flat_size = self[flat_indices].shape.unwrap_1();

        let result_flat = if let Some(index_f) = self.as_single_const(indices) {
            // replace gather with simpler slice (+ repeat) operator
            let index = index_f as usize;
            assert_eq!(index as f32, index_f, "Index must be an integer, got {}", index_f);

            let result_flat_single = self.slice(input, axis, SliceRange::single(index));
            let result_flat = self.repeat(result_flat_single, axis, flat_size);

            assert_eq!(self[result_flat].shape, result_shape_flat);
            result_flat
        } else {
            // do a full gather operation
            self.push(
                result_shape_flat,
                Operation::Gather {
                    input,
                    axis,
                    indices: flat_indices,
                },
            )
        };

        let result = self.view(result_flat, result_shape);
        result
    }

    /// Concatenate `inputs` along `axis`.
    /// `base_shape` can be provided to allow the result shape to be inferred in case `inputs` is empty.
    #[must_use]
    pub fn concat(&mut self, inputs: Vec<Value>, axis: usize, base_shape: Option<Shape>) -> Value {
        // skip operation if there is only a single input
        // TODO also remove empty inputs from the list of operands
        // TODO skip concat entirely if there is only a single nonempty input
        // TODO skip entire operation if the output is empty (generalize this to all operations?)
        if inputs.len() == 1 {
            let shape = &self[inputs[0]].shape;
            shape.assert_has_axis(axis);
            return inputs[0];
        }

        let base_shape = base_shape.unwrap_or_else(|| {
            assert!(
                !inputs.is_empty(),
                "Cannot infer concatenation shape without any values"
            );
            self[inputs[0]].shape.replace(axis, shape![0])
        });

        let size_along_axis = inputs
            .iter()
            .map(|&v| {
                assert_eq!(
                    self[v].shape.replace(axis, shape![0]),
                    base_shape,
                    "All concatenated values must match base shape on non-concatenated axes"
                );
                self[v].shape.dims[axis]
            })
            .sum::<Option<Size>>()
            .unwrap_or_else(|| {
                let input_shapes = inputs.iter().map(|&v| &self[v].shape).collect_vec();
                panic!("Could not add all concatenation sizes: {:?}", input_shapes);
            });

        let result_shape = base_shape.replace(axis, shape![size_along_axis]);
        self.push(result_shape, Operation::Concat { inputs, axis })
    }

    /// Apply 2D convolution.
    #[must_use]
    pub fn conv(
        &mut self,
        input: Value,
        filter: Value,
        stride_y: usize,
        stride_x: usize,
        padding_y: usize,
        padding_x: usize,
    ) -> Value {
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

        let padded_input_h = input_h + 2 * padding_y;
        let padded_input_w = input_w + 2 * padding_x;
        assert!(
            padded_input_h >= kernel_h && padded_input_w >= kernel_w,
            "Kernel must fit inside of padded input"
        );

        // operations are ordered to avoid underflow
        let output_h = (padded_input_h - (kernel_h - 1) - 1) / stride_y + 1;
        let output_w = (padded_input_w - (kernel_w - 1) - 1) / stride_x + 1;
        let output_shape = shape![batch_size, output_channels, output_h, output_w];

        let details = ConvDetails {
            batch_size,
            input_channels,
            output_channels,
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            stride_y,
            stride_x,
            padding_y,
            padding_x,
            output_h,
            output_w,
        };
        self.push(output_shape, Operation::Conv { input, details, filter })
    }

    /// Apply a linear transformation.
    /// Input shape `[b, Ci]` and weight shape `[Co, Ci]` result in an output with shape `[b, Co]`.
    #[must_use]
    pub fn linear(&mut self, input: Value, weight: Value) -> Value {
        let weight_transposed = self.permute(weight, vec![1, 0]);
        self.mat_mul(input, weight_transposed)
    }

    /// General matrix multiply, with broadcasting.
    ///
    /// * The last two axes should have shapes `[n, p]` and `[p, m]` and will result in an output shape `[n, m]`
    /// * The preceding axes are broadcast together and reappear in the output as-is.
    #[must_use]
    pub fn mat_mul(&mut self, left: Value, right: Value) -> Value {
        let left_shape = &self[left].shape;
        let right_shape = &self[right].shape;

        assert!(
            left_shape.rank() >= 2 && right_shape.rank() >= 2,
            "Matmul operands must have rank >= 2, got shapes {} and {}",
            left_shape,
            right_shape
        );

        let (left_head, left_tail) = left_shape.split(left_shape.rank() - 2);
        let (right_head, right_tail) = right_shape.split(right_shape.rank() - 2);

        // check tails match
        let [m, n0] = left_tail.unwrap_2();
        let [n1, p] = right_tail.unwrap_2();
        assert_eq!(
            n0, n1,
            "Inner matmul dimension must match, got shapes {} and {}",
            left_shape, right_shape
        );
        let result_tail = shape![m, p];

        // broadcast heads
        let result_head = broadcast_shape_symmetric(&left_head, &right_head);
        let batch_size = result_head.size();
        let left_broadcast = self.broadcast(left, result_head.clone().concat(&left_tail));
        let right_broadcast = self.broadcast(right, result_head.clone().concat(&right_tail));

        // flatten for bmm
        let left_flat = self.view(left_broadcast, left_tail.insert(0, batch_size));
        let right_flat = self.view(right_broadcast, right_tail.insert(0, batch_size));
        let result_flat = self.batched_mat_mul(left_flat, right_flat);

        // unflatten into final shape
        let result = self.view(result_flat, result_head.concat(&result_tail));
        result
    }

    /// Batched matrix multiply, without any automatic broadcasting.
    /// Inputs must have shapes `[b, m, n]`, `[b, n, p]` and the result has shape `[b, m, p]`.
    #[must_use]
    pub fn batched_mat_mul(&mut self, left: Value, right: Value) -> Value {
        let [b0, m, n0] = self[left].shape.unwrap_3();
        let [b1, n1, p] = self[right].shape.unwrap_3();

        assert!(
            b0 == b1 && n0 == n1,
            "Batched matmul dimension mismatch, got shapes {} and {}",
            self[left].shape,
            self[right].shape
        );

        let result_shape = shape![b0, m, p];
        self.push(result_shape, Operation::MatMul { left, right })
    }

    #[must_use]
    pub fn softmax(&mut self, input: Value, axis: usize) -> Value {
        let input_shape = &self[input].shape;
        input_shape.assert_has_axis(axis);

        let new_shape = input_shape.clone();
        self.push(new_shape, Operation::Softmax { input, axis })
    }

    #[must_use]
    pub fn layernorm(&mut self, input: Value, axis: usize, eps: f32) -> Value {
        let input_shape = &self[input].shape;
        input_shape.assert_has_axis(axis);

        let new_shape = input_shape.clone();
        self.push(
            new_shape,
            Operation::Layernorm {
                input,
                axis,
                eps: Total::from(eps),
            },
        )
    }

    /// Reduce `input` along the given `axes`.
    /// The result shape is the same as the input shape but without the reduces axes.
    #[must_use]
    pub fn reduce(&mut self, input: Value, axes: Vec<usize>, op: ReduceOp) -> Value {
        if axes.is_empty() {
            return input;
        }

        let input_shape = &self[input].shape;

        // check that the axes are in bounds
        for &axis in &axes {
            input_shape.assert_has_axis(axis);
        }

        let new_shape = input_shape.replace_all(&axes, shape![]);
        self.push(new_shape, Operation::Reduce { input, axes, op })
    }

    /// Elementwise sigmoid.
    #[must_use]
    pub fn sigmoid(&mut self, input: Value) -> Value {
        self.push(
            self[input].shape.clone(),
            Operation::Unary {
                input,
                op: UnaryOp::Sigmoid,
            },
        )
    }

    /// Elementwise relu.
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
            curr = self.binary(BinaryOp::Min, curr, max_value);
        }

        if min != f32::NEG_INFINITY {
            let min_value = self.constant(right_shape, vec![min]);
            curr = self.binary(BinaryOp::Max, curr, min_value);
        }

        curr
    }

    #[must_use]
    pub fn add(&mut self, left: Value, right: Value) -> Value {
        self.binary(BinaryOp::Add, left, right)
    }

    #[must_use]
    pub fn sub(&mut self, left: Value, right: Value) -> Value {
        self.binary(BinaryOp::Sub, left, right)
    }

    #[must_use]
    pub fn mul(&mut self, left: Value, right: Value) -> Value {
        self.binary(BinaryOp::Mul, left, right)
    }

    #[must_use]
    pub fn pow(&mut self, left: Value, right: Value) -> Value {
        self.binary(BinaryOp::Pow, left, right)
    }

    // Elementwise binary operation.
    #[must_use]
    pub fn unary(&mut self, op: UnaryOp, input: Value) -> Value {
        self.push(self[input].shape.clone(), Operation::Unary { op, input })
    }

    /// Compute elementwise binary operation.
    /// Both inputs must have the same rank (or right must have rank 0), the right shape is broadcasted to the left shape.
    #[must_use]
    pub fn binary(&mut self, op: BinaryOp, left: Value, right: Value) -> Value {
        let skip = match op {
            BinaryOp::Sub | BinaryOp::Add => self.is_const_filled_with(right, 0.0),
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow => self.is_const_filled_with(right, 1.0),
            BinaryOp::Min => self.is_const_filled_with(right, f32::INFINITY),
            BinaryOp::Max => self.is_const_filled_with(right, f32::NEG_INFINITY),
        };
        // TODO only skip after shape checking
        //   also check other functions
        if skip {
            return left;
        }

        let result_shape = broadcast_shape_symmetric(&self[left].shape, &self[right].shape);
        let left = self.broadcast(left, result_shape.clone());
        let right = self.broadcast(right, result_shape.clone());

        self.push(result_shape, Operation::Binary { left, right, op })
    }

    /// Computes the operations described by `graph` on the given inputs.
    ///
    /// This can be used to cleanly compose multiple graphs together.
    #[must_use]
    pub fn call(&mut self, graph: &Graph, inputs: &[Value]) -> Vec<Value> {
        // check inputs
        assert_eq!(inputs.len(), graph.inputs.len(), "Wrong number of inputs");
        for (&input, &graph_input) in zip_eq(inputs, &graph.inputs) {
            assert_eq!(self[input].shape, graph[graph_input].shape, "Wrong input shape");
        }

        let mut map = HashMap::new();

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

pub fn broadcast_shape_symmetric(left: &Shape, right: &Shape) -> Shape {
    let rank = max(left.rank(), right.rank());

    // pad with leading 1 axes
    let left = Shape::ones(rank - left.rank()).concat(&left);
    let right = Shape::ones(rank - right.rank()).concat(&right);

    // decide the matching axes for both
    let result = zip_eq(&left.dims, &right.dims)
        .map(|(&l, &r)| match (l, r) {
            (Size::ONE, other) | (other, Size::ONE) => other,
            (any, other) if any == other => any,
            _ => panic!("Cannot broadcast {} and {} in shapes {} and {}", l, r, left, right),
        })
        .collect_vec();

    Shape::new(result)
}

impl Debug for Graph {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Graph")
            .field("inputs", &self.inputs().iter().map(|&v| &self[v].shape).collect_vec())
            .field("outputs", &self.outputs().iter().map(|&v| &self[v].shape).collect_vec())
            .finish_non_exhaustive()
    }
}

// TODO output nicer table with debug_id near the front
impl Display for Graph {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let Graph {
            check,
            values,
            new_values: _,
            inputs,
            outputs,
        } = self;

        writeln!(f, "Graph {{")?;
        writeln!(f, "  check: {},", self.check)?;

        let input_shapes = self.inputs().iter().map(|&v| &self[v].shape).collect_vec();
        let output_shapes = self.outputs().iter().map(|&v| &self[v].shape).collect_vec();
        writeln!(f, "  input_shapes: {:?},", input_shapes)?;
        writeln!(f, "  output_shapes: {:?},", output_shapes)?;
        writeln!(f, "  inputs: {:?},", inputs)?;
        writeln!(f, "  outputs: {:?},", outputs)?;

        writeln!(f, "  values: [")?;
        for (i, info) in values.iter().enumerate() {
            writeln!(
                f,
                "    {:?} = {:?},",
                Value {
                    index: i,
                    check: *check,
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

impl Display for SliceRange {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.step == 1 {
            write!(f, "{}:{}", self.start, self.end)
        } else {
            write!(f, "{}:{}:{}", self.start, self.end, self.step)
        }
    }
}

impl From<std::ops::Range<usize>> for SliceRange {
    fn from(range: std::ops::Range<usize>) -> Self {
        let std::ops::Range { start, end } = range;
        SliceRange::simple(start, end)
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

    pub fn assert_in_bounds(self, size: usize) {
        self.assert_valid();

        assert!(
            self.start == self.end || (self.start < size && self.end - (self.step - 1) <= size),
            "{:?} out of bounds for axis of size {}",
            self,
            size
        )
    }
}

impl UnaryOp {
    pub const ALL: &'static [Self] = &[
        UnaryOp::Abs,
        UnaryOp::Neg,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Sqrt,
        UnaryOp::Sigmoid,
        UnaryOp::Tanh,
        UnaryOp::Erf,
        UnaryOp::Mish,
    ];

    pub fn map(self, x: f32) -> f32 {
        match self {
            UnaryOp::Abs => x.abs(),
            UnaryOp::Neg => -x,
            UnaryOp::Sin => x.sin(),
            UnaryOp::Cos => x.cos(),
            UnaryOp::Exp => x.exp(),
            UnaryOp::Log => x.ln(),
            UnaryOp::Sqrt => x.sqrt(),
            UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            UnaryOp::Tanh => x.tanh(),
            UnaryOp::Erf => erf(x),
            UnaryOp::Mish => x * (1.0 + x.exp()).ln().tanh(),
        }
    }
}

impl BinaryOp {
    pub const ALL: &'static [Self] = &[
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Pow,
        BinaryOp::Min,
        BinaryOp::Max,
    ];

    pub fn map(self, left: f32, right: f32) -> f32 {
        match self {
            BinaryOp::Add => left + right,
            BinaryOp::Sub => left - right,
            BinaryOp::Mul => left * right,
            BinaryOp::Div => left / right,
            BinaryOp::Pow => f32::powf(left, right),
            BinaryOp::Min => f32::min(left, right),
            BinaryOp::Max => f32::max(left, right),
        }
    }
}

impl ReduceOp {
    pub const ALL: &'static [Self] = &[
        ReduceOp::Sum,
        ReduceOp::Mean,
        ReduceOp::Prod,
        ReduceOp::Min,
        ReduceOp::Max,
    ];

    pub fn identity(self) -> f32 {
        match self {
            ReduceOp::Sum | ReduceOp::Mean => 0.0,
            ReduceOp::Prod => 1.0,
            ReduceOp::Min => f32::INFINITY,
            ReduceOp::Max => f32::NEG_INFINITY,
        }
    }

    pub fn operation(self) -> (BinaryOp, bool) {
        match self {
            ReduceOp::Sum => (BinaryOp::Add, false),
            ReduceOp::Mean => (BinaryOp::Add, true),
            ReduceOp::Prod => (BinaryOp::Mul, false),
            ReduceOp::Min => (BinaryOp::Min, false),
            ReduceOp::Max => (BinaryOp::Max, false),
        }
    }

    pub fn reduce(self, seq: impl IntoIterator<Item = f32>) -> f32 {
        let (op, is_mean) = self.operation();

        let mut count = 0;
        let total = seq.into_iter().fold(self.identity(), |acc, x| {
            count += 1;
            op.map(acc, x)
        });

        if is_mean {
            total / count as f32
        } else {
            total
        }
    }
}

/// Formula and coefficients from <https://en.wikipedia.org/wiki/Error_function#Numerical_approximations>
/// (Abramowitz and Stegun),
/// Max error `3e-7`. We use f64 internally to ensure there are no additional errors introduced.
pub fn erf(x: f32) -> f32 {
    let sign = x.signum();
    let x_abs = x.abs() as f64;

    const A: &[f64] = &[
        1.0,
        0.0705230784,
        0.0422820123,
        0.0092705272,
        0.0001520143,
        0.0002765672,
        0.0000430638,
    ];

    let d: f64 = A
        .iter()
        .copied()
        .enumerate()
        .map(|(i, a)| a * x_abs.powi(i as i32))
        .sum();
    let y_abs = 1.0 - 1.0 / d.powi(16);

    return sign * y_abs as f32;
}
