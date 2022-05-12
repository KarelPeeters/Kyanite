use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use crate::graph::{BinaryOp, Graph, Operation, Value};
use crate::optimizer::OptimizerSettings;

#[derive(Debug)]
pub struct Optimizer<'a> {
    settings: OptimizerSettings,

    pub old_graph: &'a Graph,
    pub new_graph: Graph,

    fuse_candidates: HashSet<Value>,
    mapping: HashMap<Value, Value>,
}

impl<'a> Optimizer<'a> {
    pub fn new(settings: OptimizerSettings, old_graph: &'a Graph) -> Self {
        let fuse_candidates = find_hidden_values_used_once(old_graph).collect();

        Optimizer {
            settings,
            fuse_candidates,
            new_graph: Graph::new(),
            old_graph,
            mapping: HashMap::default(),
        }
    }

    pub fn define(&mut self, old: Value, new: Value) {
        let prev = self.mapping.insert(old, new);
        assert!(prev.is_none());
    }

    pub fn map(&mut self, old_value: Value) -> Value {
        if let Some(new_value) = self.mapping.get(&old_value).copied() {
            return new_value;
        }

        let new_value = self.map_new(old_value);
        self.define(old_value, new_value);
        new_value
    }

    fn map_new(&mut self, old_value: Value) -> Value {
        // try fusing the value
        if let Some(fused) = self.try_fuse(old_value) {
            return fused;
        }

        // fallback, copy the old operation
        let shape = self.old_graph[old_value].shape.clone();
        let old_operation = &self.old_graph[old_value].operation;
        let new_operation = old_operation.clone_map_inputs(|old_input| self.map(old_input));
        self.new_graph.push(shape, new_operation)
    }

    fn try_fuse(&mut self, old_start: Value) -> Option<Value> {
        if let Some(result) = self.try_fuse_clamp(old_start) {
            return Some(result);
        }
        if let Some(result) = self.try_fuse_conv_affine(old_start) {
            return Some(result);
        }
        if let Some(result) = self.try_convert_div_to_mul(old_start) {
            return Some(result);
        }

        None
    }

    /// Fuse _multiple_ sequential min and max operations into a single min and max operation.
    fn try_fuse_clamp(&mut self, old_start: Value) -> Option<Value> {
        let mut total_min = f32::NEG_INFINITY;
        let mut total_max = f32::INFINITY;

        let old_input = self.follow_if(old_start, |_, _, operation| {
            if let &Operation::Binary {
                left: old_left,
                right: old_right,
                op: op @ (BinaryOp::Min | BinaryOp::Max),
            } = operation
            {
                // if right is a single constant value we can fuse it
                if let Some(value) = self.old_graph.as_const(old_right) {
                    if value.len() == 1 {
                        let &f = value.iter().next().unwrap();

                        match op {
                            BinaryOp::Min => total_max = f32::min(total_max, f),
                            BinaryOp::Max => total_min = f32::max(total_min, f),
                            _ => unreachable!(),
                        }
                        return Some(old_left);
                    }
                }
            }
            None
        })?;

        let new_input = self.map(old_input);
        let new_output = self.new_graph.clamp(new_input, total_min, total_max);
        Some(new_output)
    }

    // TODO also get this to work for 1D convolutions
    fn try_fuse_conv_affine(&mut self, old_start: Value) -> Option<Value> {
        let group = self.try_build_affine_group(old_start)?;

        let new_input = self.map(group.old_input());
        let new_start = group.apply_fused(self.settings, &mut self.new_graph, new_input);

        Some(new_start)
    }

    fn try_convert_div_to_mul(&mut self, old_start: Value) -> Option<Value> {
        if let &Operation::Binary {
            left,
            right,
            op: BinaryOp::Div,
        } = &self.old_graph[old_start].operation
        {
            if let Some(data) = self.old_graph.as_const(right) {
                let new_data = data.iter().map(|&x| 1.0 / x).collect_vec();
                let new_right = self.new_graph.constant(self.old_graph[right].shape.clone(), new_data);

                let new_left = self.map(left);
                let result = self.new_graph.mul(new_left, new_right);
                Some(result)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn follow_if(
        &self,
        start: Value,
        mut next: impl FnMut(&Graph, Value, &Operation) -> Option<Value>,
    ) -> Option<Value> {
        let mut curr = start;

        loop {
            if !self.fuse_candidates.contains(&curr) {
                break;
            }

            if let Some(next) = next(self.old_graph, curr, &self.old_graph[curr].operation) {
                curr = next;
            } else {
                break;
            }
        }

        if curr == start {
            None
        } else {
            Some(curr)
        }
    }
}

pub fn values_use_count<'a>(graph: &'a Graph) -> impl Iterator<Item = (Value, usize)> + 'a {
    let all_operands = graph.values().flat_map(|v| graph[v].operation.inputs()).collect_vec();

    graph.values().map(move |value| {
        let operand_count = all_operands.iter().filter(|&&other| other == value).count();
        let output_count = graph.outputs().iter().filter(|&&other| other == value).count();

        let use_count = operand_count + output_count;
        (value, use_count)
    })
}

pub fn find_hidden_values_used_once<'a>(graph: &'a Graph) -> impl Iterator<Item = Value> + 'a {
    values_use_count(graph)
        .filter(move |&(value, use_count)| {
            use_count == 1 && !graph.inputs().contains(&value) && !graph.outputs().contains(&value)
        })
        .map(|(v, _)| v)
}
