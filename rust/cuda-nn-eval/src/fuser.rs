use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Index;

use crate::graph::{ConvShape, Graph, Operation, Value};

//TODO implement a proper scheduler, right now we're basically abusing the visit order as a topological sort
//TODO in general this thing is hard to change, it would be better if we had a general graph matching thing
pub struct FusedGraph {
    schedule: Vec<FusedValueInfo>,
    visited: HashMap<Value, FusedValue>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct FusedValue(usize);

#[derive(Debug, Copy, Clone)]
pub enum Activation {
    Linear,
    Relu,
}

#[derive(Debug, Copy, Clone)]
pub enum FusedValueInfo {
    Input(Value),
    Constant(Value),
    /// A single operation representing `act(conv(input, filter) + bias + res)`
    FusedOperation {
        value: Value,

        input: FusedValue,
        input_shape_view: [i32; 4],

        filter: FusedValue,
        conv_shape: ConvShape,
        bias: FusedValue,
        res_input: Option<FusedValue>,
        act_mode: Activation,
    },
}

impl FusedValueInfo {
    pub fn value(self) -> Value {
        let (
            FusedValueInfo::Input(value) |
            FusedValueInfo::Constant(value) |
            FusedValueInfo::FusedOperation { value, .. }
        ) = self;
        value
    }
}

// TODO this whole thing is hard to change, and add more patterns too in the future
//   it would be better if we had some general-purpose graph matching engine here
impl FusedGraph {
    pub fn new(graph: &Graph) -> Self {
        let mut result = FusedGraph {
            schedule: Default::default(),
            visited: Default::default(),
        };

        for &output in graph.outputs() {
            result.visit(graph, output);
        }

        result
    }
    
    pub fn find(&self, value: Value) -> FusedValue {
        *self.visited.get(&value)
            .unwrap_or_else(|| panic!("Could not find value {:?}, maybe it was fused away", value))
    }

    pub fn schedule(&self) -> impl Iterator<Item=FusedValue> {
        (0..self.schedule.len()).map(FusedValue)
    }

    /// Append an operation to the schedule.
    fn push(&mut self, value: Value, info: FusedValueInfo) -> FusedValue {
        let index = self.schedule.len();
        self.schedule.push(info);
        let fused_value = FusedValue(index);

        let prev = self.visited.insert(value, fused_value);
        if let Some(prev) = prev {
            let prev_info = &self[prev];
            panic!(
                "Trying to add {:?} = {:?}, {:?}\n  but we already have = {:?}, {:?}",
                value, fused_value, info, prev, prev_info
            )
        }

        fused_value
    }

    fn visit(&mut self, g: &Graph, value: Value) -> FusedValue {
        // immediately return if we've already visited this value
        if let Some(&fused_value) = self.visited.get(&value) {
            return fused_value;
        }

        // if it's an input or constant just wrap it
        if let Operation::Input = g[value].operation {
            return self.push(value, FusedValueInfo::Input(value));
        }
        if let Operation::Constant { .. } = g[value].operation {
            return self.push(value, FusedValueInfo::Constant(value));
        }

        self.visit_fuse(g, value)
    }

    fn visit_fuse(&mut self, g: &Graph, value: Value) -> FusedValue {
        let mut next = value;

        // accept activation
        let act_mode = if let Operation::Relu { input } = g[next].operation {
            next = input;
            Activation::Relu
        } else {
            Activation::Linear
        };

        // accept residual
        if let Operation::Add { left, right } = g[next].operation {
            // pick the first input that we can fuse
            let fused =
                if let Some(fused) = self.try_finish_fuse(g, value, left, Some(right), act_mode) {
                    fused
                } else if let Some(fused) = self.try_finish_fuse(g, value, right, Some(left), act_mode) {
                    fused
                } else {
                    panic!("Failed to finish fusing both {:?} and {:?}", left, right);
                };

            return fused;
        };

        // otherwise just continue fusing
        self.try_finish_fuse(g, value, next, None, act_mode)
            .unwrap_or_else(|| panic!("Failed to finish fusing {:?}", next))
    }

    // this function doesn't change internal state if the fusing failed.
    fn try_finish_fuse(
        &mut self,
        g: &Graph,
        value: Value,
        next: Value,
        res_input: Option<Value>,
        act_mode: Activation,
    ) -> Option<FusedValue> {
        // require bias
        //TODO it's not that hard to remove this requirement
        let (bias_input, bias) = if let Operation::Bias { input, bias } = g[next].operation {
            (input, self.visit(g, bias))
        } else {
            return None;
        };

        // require convolution
        let (conv_input, filter, conv_shape) =
            if let &Operation::Conv { input, filter, conv_shape } = &g[bias_input].operation {
                (input, self.visit(g, filter), conv_shape)
            } else {
                return None;
            };

        // accept flatten
        let (final_input, input_shape_view) = if let Operation::Flatten { input } = g[conv_input].operation {
            let [n, c, h, w] = g[input].shape;
            (input, [n, c * h * w, 1, 1])
        } else {
            (conv_input, g[conv_input].shape)
        };

        let final_input_fused = self.visit(g, final_input);
        let res_input_fused = res_input.map(|res_input| self.visit(g, res_input));

        let operation = FusedValueInfo::FusedOperation {
            value: next,
            input: final_input_fused,
            input_shape_view,
            res_input: res_input_fused,
            bias,
            filter,
            conv_shape,
            act_mode,
        };

        Some(self.push(value, operation))
    }
}

impl Index<Value> for FusedGraph {
    type Output = FusedValue;

    fn index(&self, index: Value) -> &Self::Output {
        self.visited.get(&index)
            .unwrap_or_else(|| panic!("{:?} not found, maybe it was fused away", index))
    }
}

impl Index<FusedValue> for FusedGraph {
    type Output = FusedValueInfo;

    fn index(&self, index: FusedValue) -> &Self::Output {
        assert!(index.0 < self.schedule.len(), "Value {:?} does not belong to this graph", index);
        &self.schedule[index.0]
    }
}

impl Debug for FusedGraph {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let FusedGraph { schedule, visited: _, } = self;

        writeln!(f, "FusedGraph {{")?;

        writeln!(f, "  schedule: [")?;
        for (i, info) in schedule.iter().enumerate() {
            writeln!(f, "    {:?} -> {:?},", FusedValue(i), info)?;
        }
        writeln!(f, "  ],")?;

        writeln!(f, "}}")?;
        Ok(())
    }
}
