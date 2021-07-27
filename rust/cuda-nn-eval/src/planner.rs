use std::collections::HashMap;
use std::ops::Index;

use cuda_sys::bindings::cudnnActivationMode_t;

use crate::graph::{Graph, Operation, Value};
use std::fmt::{Debug, Formatter};

pub struct FusedGraph {
    //TODO implement a proper scheduler, right now we're basically abusing the visit order as a topological sort
    schedule: Vec<FusedValueInfo>,
    visited: HashMap<Value, Option<FusedValue>>,

    // only used during construction
    next_filter_index: usize,
    next_bias_index: usize,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct FusedValue(usize);

#[derive(Debug, Copy, Clone)]
pub enum FusedValueInfo {
    Input(Value),
    FusedOperation {
        value: Value,

        input_fused: FusedValue,
        res_input: Option<FusedValue>,

        output_channels: i32,
        kernel_size: i32,
        padding: i32,

        filter_index: usize,
        bias_index: usize,
        act_mode: cudnnActivationMode_t,
    },
}

impl FusedValueInfo {
    pub fn value(self) -> Value {
        let (FusedValueInfo::Input(value) | FusedValueInfo::FusedOperation { value, .. }) = self;
        value
    }

    pub fn uses_input(&self, value: FusedValue) -> bool {
        match self {
            FusedValueInfo::Input(_) =>
                false,
            &FusedValueInfo::FusedOperation { input_fused, res_input: res, .. } =>
                input_fused == value || res == Some(value)
        }
    }
}

impl FusedGraph {
    pub fn new(graph: &Graph) -> Self {
        let mut result = FusedGraph {
            schedule: Default::default(),
            visited: Default::default(),
            next_filter_index: 0,
            next_bias_index: 0,
        };

        for &output in graph.outputs() {
            result.visit(graph, output);
        }

        result
    }

    pub fn schedule(&self) -> impl Iterator<Item=FusedValue> {
        (0..self.schedule.len()).map(FusedValue)
    }

    fn push(&mut self, info: FusedValueInfo) -> FusedValue {
        let index = self.schedule.len();
        self.schedule.push(info);
        FusedValue(index)
    }

    fn visit(&mut self, g: &Graph, value: Value) -> FusedValue {
        // immediately return if we've already visited this value
        if let Some(fused_value) = self.visited.get(&value) {
            match *fused_value {
                None => panic!("Limitation: {:?} was fused away and cannot be used any more", value),
                Some(fused_value) => return fused_value,
            }
        }

        //if it's an input just wrap it
        if let Operation::Input = g[value].operation {
            return self.push(FusedValueInfo::Input(value));
        }

        // all of the values involved in this fused operation, including input and output
        let mut intermediate_values = vec![value];
        let mut inner_value = value;

        // accept activation
        let act_mode = if let Operation::Relu { input } = g[inner_value].operation {
            intermediate_values.push(input);
            inner_value = input;
            cudnnActivationMode_t::CUDNN_ACTIVATION_RELU
        } else {
            cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY
        };

        // accept residual
        //TODO this left/right hardcoding is very limiting and brittle
        let res_left = if let Operation::Add { left, right } = g[inner_value].operation {
            intermediate_values.push(right);
            inner_value = right;
            Some(left)
        } else {
            None
        };

        // require bias
        if let Operation::Bias { input, .. } = g[inner_value].operation {
            intermediate_values.push(input);
            inner_value = input;
        } else {
            panic!("Limitation: Expected bias operation, got {:?}", g[value].operation);
        };

        // require convolution
        let (
            output_channels,
            kernel_size,
            padding
        ) = if let Operation::Conv { input, output_channels, kernel_size, padding, flat_weights: _ } = g[inner_value].operation {
            intermediate_values.push(input);
            inner_value = input;
            (
                output_channels,
                kernel_size,
                padding
            )
        } else {
            panic!("Limitation: Expected conv operation, got {:?}", g[value].operation);
        };

        // mark all intermediate values as fused, skipping the first and last
        for &fused_value in &intermediate_values[1..intermediate_values.len() - 1] {
            if let Some(prev) = self.visited.insert(fused_value, None) {
                panic!("fused {:?} already visited with mapping {:?}", fused_value, prev);
            }
        }

        let input_fused = self.visit(g, inner_value);

        // only get indices after the previous operations have been visited
        let filter_index = post_inc(&mut self.next_filter_index);
        let bias_index = post_inc(&mut self.next_bias_index);

        let res_input = res_left.map(|left| {
            self.visited.get(&left)
                .unwrap_or_else(|| panic!("Limitation: left {:?} should have been visited already", value))
                .unwrap_or_else(|| panic!("Limitation: left {:?} was fused away", value))
        });

        let operation = FusedValueInfo::FusedOperation {
            value,
            input_fused,
            res_input,
            output_channels,
            kernel_size,
            padding,
            filter_index,
            bias_index,
            act_mode,
        };
        let value_fused = self.push(operation);

        if let Some(prev) = self.visited.insert(value, Some(value_fused)) {
            panic!("non-fused {:?} already visited with mapping {:?}", value, prev);
        };

        value_fused
    }
}

impl Index<Value> for FusedGraph {
    type Output = FusedValue;

    fn index(&self, index: Value) -> &Self::Output {
        self.visited.get(&index)
            .unwrap_or_else(|| panic!("{:?} not found", index))
            .as_ref()
            .unwrap_or_else(|| panic!("Limitation: {:?} was fused away", index))
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
        let FusedGraph { schedule, visited: _, next_filter_index: _, next_bias_index: _ } = self;

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

fn post_inc(x: &mut usize) -> usize {
    let t = *x;
    *x += 1;
    t
}
