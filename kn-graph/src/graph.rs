use std::cmp::max;
use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt::{Debug, Display, Formatter};
use std::ops::Index;

use decorum::Total;
use itertools::{zip_eq, Itertools};
use ndarray::{ArrayView, IxDyn};
use rand::random;

use crate::cpu::{run_cpu_const_operation, OperationError, OperationResult};
use crate::dtype::{dispatch_dtensor, dispatch_dtype, map_dscalar_pair, DScalar, DTensor, DType, IntoDScalar, Tensor};
use crate::optimizer::recurse::heap_recurse;
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
/// # use kn_graph::dtype::DType;
/// use kn_graph::graph::*;
/// # use kn_graph::shape;
/// # use kn_graph::shape::*;
/// // create a new graph
/// let mut graph = Graph::new();
///
/// // define the inputs
/// let x = graph.input(shape![Size::BATCH, 4, 8, 8], DType::F32);
///
/// // define constants
/// let w_data = vec![0.5; 4 * 4 * 3 * 3];
/// let w = graph.constant::<f32>(shape![4, 4, 3, 3], w_data);
/// let b_data = vec![0.5; 4];
/// let b = graph.constant::<f32>(shape![4, 1, 1], b_data);
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
// TODO override clone manually, replace check value
// TODO think about two builder categories:
//     * things that map directly to an operation, with all the type and shape checking
//     * things that do optimizations, extra broadcasting, ...
//   alternatively do extra checking in `self.push`?
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
    pub dtype: DType,
    pub operation: Operation,
    pub debug_id: String,
    non_output_uses: usize,
}

/// The core set of graph operations.
/// Some attempt was made to keep operations orthogonal but flexible, so they can be composed easily.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Operation {
    /// A runtime-variable input.
    Input { index: usize },
    /// A constant built into the network.
    Constant { tensor: DTensor },

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
    // TODO "select"/"where" operation
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

    /// Cast to a different type.
    /// When possible the value is preserved or at least approximated.
    ValueCast(DType),
    /// Cast to a different type.
    /// The bit pattern is kept, so the value is not necessarily preserved.
    /// The type before and after the cast must have the same size.
    BitCast(DType),
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
    // TODO remove mean and rely on operator fusion instead
    //   definitely do this, it's getting pretty ugly in the planner
    Mean,
    Prod,
    Max,
    Min,
}

impl Operation {
    pub fn inputs(&self) -> Vec<Value> {
        match self {
            Operation::Input { index: _ } => vec![],
            Operation::Constant { tensor: _ } => vec![],
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
            &Operation::Constant { ref tensor } => Operation::Constant { tensor: tensor.clone() },
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
    pub dtype: DType,
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

    pub fn shape_dtype(&self, value: Value) -> (&Shape, DType) {
        let info = &self[value];
        (&info.shape, info.dtype)
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
    pub fn as_const(&self, value: Value) -> Option<DTensor> {
        // we have to use heap_recurse to avoid stack overflows

        // TODO always immediately evaluate all possible values instead?
        // TODO store this cache in the graph permanently?
        //   this will all take a bunch of memory and time :(
        let mut cache: HashMap<Value, OperationResult> = HashMap::new();

        let f_cached = |curr| {
            let mut missing_arg = None;

            let res = run_cpu_const_operation(&self[curr], |arg| {
                match cache.get(&arg) {
                    // already evaluated
                    Some(Ok(tensor)) => Ok(tensor.clone()),
                    Some(&Err(err)) => Err(err),
                    // not evaluated yet, bubble back to the top
                    None => {
                        missing_arg = Some(arg);
                        //   the exact error used here doesn't matter
                        Err(OperationError::MissingOperand)
                    }
                }
            });

            // continue bubbling
            if let Some(missing_arg) = missing_arg {
                assert_eq!(res, Err(OperationError::MissingOperand));
                return Err(missing_arg);
            }

            let prev = cache.insert(curr, res.clone());
            assert!(prev.is_none());

            Ok(res)
        };

        let res = heap_recurse(value, f_cached);
        res.ok()
    }

    /// Returns whether `value` is effectively a constant with every element equal to `expected`.
    pub fn is_const_filled_with(&self, value: Value, expected: DScalar) -> bool {
        self.as_single_const(value).map_or(false, |actual| expected == actual)
    }

    pub fn is_const_zero(&self, value: Value) -> bool {
        self.is_const_filled_with(value, self[value].dtype.specials().zero)
    }

    pub fn is_const_one(&self, value: Value) -> bool {
        self.is_const_filled_with(value, self[value].dtype.specials().one)
    }

    /// Returns `Some(f)` if `value` is effectively a constant with every element equal to `f`.
    pub fn as_single_const(&self, value: Value) -> Option<DScalar> {
        let info = &self[value];

        match info.operation {
            Operation::Input { .. } => None,
            Operation::Constant { ref tensor } => dispatch_dtensor!(tensor, |_T, _f, tensor| {
                let &e = tensor.iter().next()?;
                tensor.iter().all(|&d| d == e).then(|| e.to_dscalar())
            }),
            Operation::View { input } => self.as_single_const(input),
            Operation::Broadcast { input } => self.as_single_const(input),
            Operation::Permute { input, permutation: _ } => self.as_single_const(input),
            Operation::Slice {
                input,
                axis: _,
                range: _,
            } => self.as_single_const(input),
            Operation::Flip { input, axis: _ } => self.as_single_const(input),
            Operation::Gather {
                input,
                axis: _,
                indices: _,
            } => self.as_single_const(input),
            Operation::Concat { ref inputs, axis: _ } => {
                let f = self.as_single_const(*inputs.first()?)?;
                inputs.iter().all(|&x| self.is_const_filled_with(x, f)).then(|| f)
            }
            Operation::Unary { input, op } => Some(op.map(self.as_single_const(input)?)),
            Operation::Binary { left, right, op } => {
                Some(op.map(self.as_single_const(left)?, self.as_single_const(right)?))
            }
            Operation::Conv { .. }
            | Operation::MatMul { .. }
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
    pub(crate) fn push(&mut self, shape: Shape, dtype: DType, operation: Operation) -> Value {
        // TODO replace const computations, especially for simple ops like unary and binary?

        let info = ValueInfo {
            shape,
            dtype,
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
    pub fn input(&mut self, shape: Shape, dtype: DType) -> Value {
        let index = self.inputs.len();
        let value = self.push(shape, dtype, Operation::Input { index });
        self.inputs.push(value);
        value
    }

    #[must_use]
    pub fn constant_tensor(&mut self, tensor: DTensor) -> Value {
        let shape = Shape::fixed(tensor.shape());
        self.push(shape, tensor.dtype(), Operation::Constant { tensor })
    }

    #[must_use]
    pub fn constant<T: IntoDScalar>(&mut self, shape: Shape, data: Vec<T>) -> Value {
        let linear = T::vec_to_dtensor(data);
        let shape = shape.unwrap_fixed("constant shape");
        let tensor = linear.reshape(shape.dims.as_slice());
        self.constant_tensor(tensor)
    }

    #[must_use]
    pub fn scalar_dyn(&mut self, value: DScalar) -> Value {
        self.constant_tensor(value.to_tensor())
    }

    #[must_use]
    pub fn scalar<T: IntoDScalar>(&mut self, value: T) -> Value {
        self.scalar_dyn(value.to_dscalar())
    }

    /// View an existing value as a new shape.
    #[must_use]
    pub fn view(&mut self, input: Value, new_shape: Shape) -> Value {
        let (input_shape, dtype) = self.shape_dtype(input);
        if &new_shape == input_shape {
            return input;
        }

        assert_eq!(
            input_shape.size(),
            new_shape.size(),
            "New shape {:?} must have the same size as old shape {:?}",
            new_shape,
            input_shape,
        );

        // only keep the last view operation
        let inner_input = if let &Operation::View { input: inner_input } = &self[input].operation {
            inner_input
        } else {
            input
        };

        self.push(new_shape, dtype, Operation::View { input: inner_input })
    }

    /// Broadcast the `input` towards `new_shape`.
    /// Additional unit axes are are inserted at the front and unit axes are repeated as necessary.
    #[must_use]
    pub fn broadcast(&mut self, input: Value, new_shape: Shape) -> Value {
        let (input_shape, dtype) = self.shape_dtype(input);
        let input_shape = input_shape.clone();

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
        self.push(new_shape, dtype, Operation::Broadcast { input: curr })
    }

    pub fn repeat_unary(&mut self, input: Value, axis: usize, count: Size) -> Value {
        let (input_shape, dtype) = self.shape_dtype(input);

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

        let new_shape = input_shape.replace(axis, shape![count]);
        self.push(new_shape, dtype, Operation::Broadcast { input })
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
        let input_info = &self[input];
        let input_shape = &input_info.shape;

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
            input_info.dtype,
            Operation::Permute {
                input: inner_input,
                permutation: full_permutation,
            },
        )
    }

    /// Slice a value along an axis.
    #[must_use]
    pub fn slice(&mut self, input: Value, axis: usize, range: SliceRange) -> Value {
        let input_info = &self[input];
        let input_shape = &input_info.shape;

        input_shape.assert_has_axis(axis);

        let input_size = input_shape.dims[axis].unwrap_fixed("Slice axis length");
        range.assert_in_bounds(input_size);
        let new_size = (range.end - range.start) / range.step;

        // skip trivial slice
        if range == SliceRange::new(0, input_size, 1) {
            return input;
        }

        let new_shape = input_shape.replace(axis, shape![new_size]);
        self.push(new_shape, input_info.dtype, Operation::Slice { input, axis, range })
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
        let input_info = &self[input];
        let input_shape = input_info.shape.clone();

        input_shape.assert_has_axis(axis);

        self.push(input_shape, input_info.dtype, Operation::Flip { input, axis })
    }

    /// Repeat `input` along a given `axis`, `count` times.
    ///
    /// This starts by emitting the entire tensor before repeating elements,
    /// similar to `torch.repeat` or `numpy.tile`.
    /// See also [repeat_interleave](Self::repeat_interleave).
    pub fn repeat(&mut self, input: Value, axis: usize, count: Size) -> Value {
        self.repeat_impl(input, axis, count, false)
    }

    /// Repeat elements of `input` along a given `axis`, `count` times.
    ///
    /// This starts by repeat each element before going to the next one,
    /// similar to `torch.repeat_interleave` or `numpy.repeat`.
    /// See also [repeat](Self::repeat).
    pub fn repeat_interleave(&mut self, input: Value, axis: usize, count: Size) -> Value {
        self.repeat_impl(input, axis, count, true)
    }

    fn repeat_impl(&mut self, input: Value, axis: usize, count: Size, inner: bool) -> Value {
        let input_shape = self[input].shape.clone();
        input_shape.assert_has_axis(axis);

        // do simpler repeat operation instead
        // TODO would this not fuse away automatically?
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
        let (input_shape, dtype) = self.shape_dtype(input);
        let (indices_shape, indices_dtype) = self.shape_dtype(indices);

        input_shape.assert_has_axis(axis);
        assert!(
            indices_dtype.is_int(),
            "Indices must be integers, got {:?}",
            indices_dtype
        );

        let result_shape = input_shape.replace(axis, indices_shape.clone());
        let result_shape_flat = input_shape.replace(axis, shape![indices_shape.size()]);

        // we support arbitrary rank indices here, but the actual operation does not
        let flat_indices = self.flatten(indices, 0);
        let flat_size = self[flat_indices].shape.unwrap_1();

        let result_flat = if let Some(index) = self.as_single_const(indices) {
            // replace gather with simpler slice + repeat operators
            let index: usize = index.unwrap_int().unwrap().try_into().unwrap();

            let result_flat_single = self.slice(input, axis, SliceRange::single(index));
            let result_flat = self.repeat(result_flat_single, axis, flat_size);

            assert_eq!(self[result_flat].shape, result_shape_flat);
            result_flat
        } else {
            // do a full gather operation
            self.push(
                result_shape_flat,
                dtype,
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
    pub fn concat(
        &mut self,
        inputs: Vec<Value>,
        axis: usize,
        base_shape: Option<Shape>,
        dtype: Option<DType>,
    ) -> Value {
        // TODO also remove empty inputs from the list of operands
        // TODO skip concat entirely if there is only a single nonempty input
        // TODO skip entire operation if the output is empty (generalize this to all operations?)

        let base_shape = base_shape.unwrap_or_else(|| {
            assert!(
                !inputs.is_empty(),
                "Cannot infer concatenation shape without any inputs"
            );
            self[inputs[0]].shape.replace(axis, shape![0])
        });
        let dtype = dtype.unwrap_or_else(|| {
            assert!(
                !inputs.is_empty(),
                "Cannot infer concatenation dtype without any inputs"
            );
            self[inputs[0]].dtype
        });

        let size_along_axis = inputs
            .iter()
            .map(|&v| {
                assert_eq!(
                    self[v].shape.replace(axis, shape![0]),
                    base_shape,
                    "All concatenated values must match base shape on non-concatenated axes"
                );
                assert_eq!(self[v].dtype, dtype, "All concatenated values must have the same dtype");
                self[v].shape.dims[axis]
            })
            .sum::<Option<Size>>()
            .unwrap_or_else(|| {
                let input_shapes = inputs.iter().map(|&v| &self[v].shape).collect_vec();
                panic!("Could not add all concatenation sizes: {:?}", input_shapes);
            });

        // skip operation if there is only a single input (only after shape and type checking)
        if inputs.len() == 1 {
            return inputs[0];
        }

        let result_shape = base_shape.replace(axis, shape![size_along_axis]);
        self.push(result_shape, dtype, Operation::Concat { inputs, axis })
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
        let (input_shape, input_dtype) = self.shape_dtype(input);
        let (filter_shape, filter_dtype) = self.shape_dtype(filter);
        assert_eq!(
            input_dtype, filter_dtype,
            "Convolution input and filter must have the same dtype"
        );
        let dtype = input_dtype;

        let [batch_size, in_c, in_h, in_w]: [Size; 4] = input_shape
            .dims
            .as_slice()
            .try_into()
            .expect("Convolution input must have rank 4");
        let [out_c, in_c_check, k_h, k_w]: [Size; 4] = filter_shape
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
            dtype,
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
        self.push(output_shape, input_dtype, Operation::Conv { input, details, filter })
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
        let (left_shape, left_dtype) = self.shape_dtype(left);
        let (right_shape, right_dtype) = self.shape_dtype(right);
        assert_eq!(left_dtype, right_dtype, "Matmul operands must have same dtype");

        let [b0, m, n0] = left_shape.unwrap_3();
        let [b1, n1, p] = right_shape.unwrap_3();

        assert!(
            b0 == b1 && n0 == n1,
            "Batched matmul dimension mismatch, got shapes {} and {}",
            left_shape,
            right_shape
        );

        let result_shape = shape![b0, m, p];
        self.push(result_shape, left_dtype, Operation::MatMul { left, right })
    }

    #[must_use]
    pub fn softmax(&mut self, input: Value, axis: usize) -> Value {
        let (input_shape, input_dtype) = self.shape_dtype(input);
        assert_eq!(input_dtype, DType::F32, "Softmax input must be f32");
        input_shape.assert_has_axis(axis);

        let new_shape = input_shape.clone();
        self.push(new_shape, input_dtype, Operation::Softmax { input, axis })
    }

    #[must_use]
    pub fn layernorm(&mut self, input: Value, axis: usize, eps: f32) -> Value {
        let (input_shape, input_dtype) = self.shape_dtype(input);
        assert_eq!(input_dtype, DType::F32, "Softmax input must be f32");
        input_shape.assert_has_axis(axis);

        let new_shape = input_shape.clone();
        self.push(
            new_shape,
            input_dtype,
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
        let (input_shape, dtype) = self.shape_dtype(input);

        // check shape and dtype
        for &axis in &axes {
            input_shape.assert_has_axis(axis);
        }
        match op {
            ReduceOp::Mean => assert_eq!(dtype, DType::F32, "Softmax input must be f32"),
            ReduceOp::Sum | ReduceOp::Prod | ReduceOp::Max | ReduceOp::Min => {}
        }

        // skip reduction
        if axes.is_empty() {
            return input;
        }

        let new_shape = input_shape.replace_all(&axes, shape![]);
        self.push(new_shape, dtype, Operation::Reduce { input, axes, op })
    }

    /// Elementwise sigmoid.
    #[must_use]
    pub fn sigmoid(&mut self, input: Value) -> Value {
        self.unary(UnaryOp::Sigmoid, input)
    }

    /// Elementwise relu.
    #[must_use]
    pub fn relu(&mut self, input: Value) -> Value {
        let (_, dtype) = self.shape_dtype(input);
        let specials = dtype.specials();
        self.clamp_dyn(input, specials.zero, specials.max)
    }

    /// Elementwise clamp.
    #[must_use]
    pub fn clamp_dyn(&mut self, input: Value, min: DScalar, max: DScalar) -> Value {
        let (_, dtype) = self.shape_dtype(input);
        assert!(
            dtype == min.dtype() && dtype == max.dtype(),
            "Clamp bounds must match value type, got min={:?} and max={:?} for {:?}",
            min,
            max,
            dtype
        );

        // careful, min/max are intentionally flipped to yield MAX(MIN(x, max), min)
        // these checks are redundant with the checks in binary, but we can skip constant allocation
        let mut curr = input;
        let specials = dtype.specials();

        if max != specials.max {
            let max_value = self.scalar_dyn(max);
            curr = self.binary(BinaryOp::Min, curr, max_value);
        }

        if min != specials.min {
            let min_value = self.scalar_dyn(min);
            curr = self.binary(BinaryOp::Max, curr, min_value);
        }

        curr
    }

    #[must_use]
    pub fn clamp<T: IntoDScalar>(&mut self, input: Value, min: T, max: T) -> Value {
        self.clamp_dyn(input, min.to_dscalar(), max.to_dscalar())
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
    pub fn unary(&mut self, op: UnaryOp, mut input: Value) -> Value {
        let (shape, input_dtype) = self.shape_dtype(input);

        let output_dtype = match op.output_dtype(input_dtype) {
            Some(d) => d,
            None => panic!("Operation {:?} not supported on dtype {:?}", op, input_dtype),
        };

        // skip cast to same type
        if let UnaryOp::ValueCast(_) | UnaryOp::BitCast(_) = op {
            if output_dtype == input_dtype {
                return input;
            }
        }

        // skip to innermost bitcast value
        // TODO skip to innermost value for exact value casts, eg. for successive truncating int casts
        //    but be careful, this is tricky stuff!
        if let UnaryOp::BitCast(_) = op {
            while let &Operation::Unary {
                op: UnaryOp::BitCast(_),
                input: inner,
            } = &self[input].operation
            {
                input = inner;
            }
        }

        self.push(shape.clone(), output_dtype, Operation::Unary { op, input })
    }

    /// Compute elementwise binary operation.
    /// Both inputs must have the same rank (or right must have rank 0), the right shape is broadcasted to the left shape.
    #[must_use]
    pub fn binary(&mut self, op: BinaryOp, left: Value, right: Value) -> Value {
        // TODO move constants to the right hand side for binary operations add/mul/min/max
        //   also think about other normalizations!
        let (left_shape, left_dtype) = self.shape_dtype(left);
        let (right_shape, right_dtype) = self.shape_dtype(right);

        let result_shape = broadcast_shape_symmetric(left_shape, right_shape);
        assert_eq!(
            left_dtype, right_dtype,
            "Binary operation {:?} requires matching dtypes, got {:?} and {:?}",
            op, left_dtype, right_dtype
        );
        let dtype = left_dtype;

        // TODO expand this skipping to be symmetric (and to do const eval if both are known and small?)
        let skip = match op {
            BinaryOp::Sub | BinaryOp::Add => self.is_const_zero(right),
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::Pow => self.is_const_one(right),
            BinaryOp::Min => self.is_const_filled_with(right, dtype.specials().max),
            BinaryOp::Max => self.is_const_filled_with(right, dtype.specials().min),
        };
        // TODO only skip after shape checking
        //   also check other functions
        if skip {
            return left;
        }

        let left = self.broadcast(left, result_shape.clone());
        let right = self.broadcast(right, result_shape.clone());

        self.push(result_shape, dtype, Operation::Binary { left, right, op })
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
                self.push(shape, graph_info.dtype, operation)
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

    // TODO variant that extracts the entire subgraph up to a given set of values, keeping the exact same inputs?
    /// Extract a small subgraph consisting of all values that go into `value`, up to a given `depth`.
    /// Values that exceed the depth are added as inputs.
    pub fn extract_subgraph(&self, value: Value, depth: u32) -> Graph {
        fn extract_impl(
            graph: &Graph,
            sub: &mut Graph,
            map: &mut HashMap<Value, Value>,
            old: Value,
            depth: u32,
        ) -> Value {
            // luckily we don't have to worry about cycles
            if let Some(&new) = map.get(&old) {
                return new;
            }

            let &ValueInfo {
                ref shape,
                dtype,
                operation: ref old_op,
                ref debug_id,
                non_output_uses: _,
            } = &graph[old];

            let new = if depth == 0 {
                // insert input
                sub.input(shape.clone(), dtype)
            } else {
                // insert operation and map operands
                let new_op = old_op.clone_map_inputs(|p| extract_impl(graph, sub, map, p, depth - 1));
                sub.push(shape.clone(), dtype, new_op)
            };

            sub.set_debug_id(new, debug_id.clone());
            let prev = map.insert(old, new);
            assert_eq!(prev, None);

            new
        }

        let mut sub = Graph::new();
        let mut map = HashMap::new();

        let new = extract_impl(self, &mut sub, &mut map, value, depth);
        sub.output(new);

        sub
    }

    /// Generate a set of dummy inputs that have the right shapes and dtypes and are all fully zero.
    /// This can be useful for some quick testing.
    pub fn dummy_zero_inputs(&self, batch_size: usize) -> Vec<DTensor> {
        // TODO add add a random version? ofc both can break gather operations, but that's acceptable
        self.inputs()
            .iter()
            .map(|&v| {
                let dtype = self[v].dtype;
                dispatch_dtype!(dtype, |_T, _fs, ft| ft(Tensor::zeros(
                    self[v].shape.eval(batch_size).dims
                )))
            })
            .collect_vec()
    }
}

/// This corresponds to [_multidimensional broadcasting_ in the ONNX spec](https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md#multidirectional-broadcasting).
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

pub fn broadcast_tensors_symmetric<'l, 'r, L, R>(
    left: &'l Tensor<L>,
    right: &'r Tensor<R>,
) -> (ArrayView<'l, L, IxDyn>, ArrayView<'r, R, IxDyn>) {
    let result_shape = broadcast_shape_symmetric(&Shape::fixed(left.shape()), &Shape::fixed(right.shape()));
    let result_shape = result_shape.as_fixed().unwrap().dims;

    let left = left.broadcast(result_shape.clone()).unwrap();
    let right = right.broadcast(result_shape).unwrap();

    (left, right)
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

// TODO switch to u64? no reason to stay stuck at u32 randomly
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

    pub fn output_dtype(self, x: DType) -> Option<DType> {
        match self {
            UnaryOp::Abs | UnaryOp::Neg => {
                if x.is_signed() {
                    Some(x)
                } else {
                    None
                }
            }
            UnaryOp::Sin
            | UnaryOp::Cos
            | UnaryOp::Exp
            | UnaryOp::Log
            | UnaryOp::Sqrt
            | UnaryOp::Sigmoid
            | UnaryOp::Tanh
            | UnaryOp::Erf
            | UnaryOp::Mish => {
                if x.is_float() {
                    Some(x)
                } else {
                    None
                }
            }
            UnaryOp::ValueCast(y) => Some(y),
            UnaryOp::BitCast(y) => {
                if x.size() == y.size() {
                    Some(y)
                } else {
                    None
                }
            }
        }
    }

    pub fn map(self, x: DScalar) -> DScalar {
        macro_rules! map_float {
            ($x:expr, |$inner:ident| $result:expr) => {{
                use $crate::dtype::{DScalar, T32, T64};
                match $x {
                    DScalar::F32(T32($inner)) => DScalar::f32($result),
                    DScalar::F64(T64($inner)) => DScalar::f64($result),
                    _ => unreachable!("Invalid dtype of {:?} for float operation {:?}", $x, self),
                }
            }};
        }
        let y = match self {
            UnaryOp::Abs => {
                assert!(x.dtype().is_signed(), "Cannot take abs of unsigned scalar");
                match x {
                    DScalar::F32(x) => DScalar::f32(x.abs()),
                    DScalar::F64(x) => DScalar::f64(x.abs()),
                    DScalar::I8(x) => DScalar::I8(x.abs()),
                    DScalar::I16(x) => DScalar::I16(x.abs()),
                    DScalar::I32(x) => DScalar::I32(x.abs()),
                    DScalar::I64(x) => DScalar::I64(x.abs()),
                    DScalar::U8(_) | DScalar::U16(_) | DScalar::U32(_) | DScalar::U64(_) | DScalar::Bool(_) => {
                        unreachable!()
                    }
                }
            }
            UnaryOp::Neg => {
                assert!(x.dtype().is_signed(), "Cannot negate unsigned scalar");
                match x {
                    DScalar::F32(x) => DScalar::f32(-*x),
                    DScalar::F64(x) => DScalar::f64(-*x),
                    DScalar::I8(x) => DScalar::I8(-x),
                    DScalar::I16(x) => DScalar::I16(-x),
                    DScalar::I32(x) => DScalar::I32(-x),
                    DScalar::I64(x) => DScalar::I64(-x),
                    DScalar::U8(_) | DScalar::U16(_) | DScalar::U32(_) | DScalar::U64(_) | DScalar::Bool(_) => {
                        unreachable!()
                    }
                }
            }
            UnaryOp::Sin => map_float!(x, |x| x.sin()),
            UnaryOp::Cos => map_float!(x, |x| x.cos()),
            UnaryOp::Exp => map_float!(x, |x| x.exp()),
            UnaryOp::Log => map_float!(x, |x| x.ln()),
            UnaryOp::Sqrt => map_float!(x, |x| x.sqrt()),
            UnaryOp::Sigmoid => map_float!(x, |x| 1.0 / (1.0 + (-x).exp())),
            UnaryOp::Tanh => map_float!(x, |x| x.tanh()),
            UnaryOp::Erf => map_float!(x, |x| erf(x as f64) as _),
            UnaryOp::Mish => map_float!(x, |x| x * (x.exp().ln_1p().tanh())),
            UnaryOp::ValueCast(to) => x.value_cast(to),
            UnaryOp::BitCast(to) => x.bit_cast(to).unwrap(),
        };

        debug_assert_eq!(self.output_dtype(x.dtype()), Some(y.dtype()));
        y
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

    pub fn map(self, left: DScalar, right: DScalar) -> DScalar {
        match self {
            BinaryOp::Add => map_dscalar_pair!(left, right, |left, right| left + right),
            BinaryOp::Sub => map_dscalar_pair!(left, right, |left, right| left - right),
            BinaryOp::Mul => map_dscalar_pair!(left, right, |left, right| left * right),
            BinaryOp::Div => map_dscalar_pair!(left, right, |left, right| left / right),
            // TODO support all types (including a mix) for pow?
            BinaryOp::Pow => DScalar::f32(left.unwrap_f32().unwrap().powf(right.unwrap_f32().unwrap())),
            BinaryOp::Min => map_dscalar_pair!(left, right, |left, right| left.min(right)),
            BinaryOp::Max => map_dscalar_pair!(left, right, |left, right| left.max(right)),
        }
    }

    pub fn map_t<T: IntoDScalar>(self, left: T, right: T) -> T {
        T::from_dscalar(self.map(left.to_dscalar(), right.to_dscalar())).unwrap()
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

    pub fn identity<T: IntoDScalar>(self) -> T {
        let specials = T::DTYPE.specials();
        let result = match self {
            ReduceOp::Sum | ReduceOp::Mean => specials.zero,
            ReduceOp::Prod => specials.one,
            ReduceOp::Min => specials.max,
            ReduceOp::Max => specials.min,
        };
        T::from_dscalar(result).unwrap()
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

    pub fn reduce_t<T: IntoDScalar>(self, seq: impl IntoIterator<Item = T>) -> T {
        let (op, is_mean) = self.operation();

        let mut count = 0;
        let total = seq.into_iter().fold(self.identity(), |acc, x| {
            count += 1;
            op.map_t(acc, x)
        });

        if is_mean {
            // TODO what to do here for non-float types?
            let total = total.to_dscalar().unwrap_f32().unwrap();
            T::from_dscalar(DScalar::f32(total / count as f32)).unwrap()
        } else {
            total
        }
    }
}

/// Formula and coefficients from <https://en.wikipedia.org/wiki/Error_function#Numerical_approximations>
/// (Abramowitz and Stegun), Max error `3e-7`.
pub fn erf(x: f64) -> f64 {
    // TODO find something that's even better for f64?
    let sign = x.signum();
    let x_abs = x.abs();

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

    sign * y_abs
}
