use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use crate::graph::{Graph, Operation, Value};
use crate::optimizer::OptimizerSettings;

pub struct Optimizer<'a> {
    settings: OptimizerSettings,

    pub old_graph: &'a Graph,
    pub new_graph: Graph,

    hidden_values: HashSet<Value>,
    mapping: HashMap<Value, Value>,
}

impl<'a> Optimizer<'a> {
    pub fn new(settings: OptimizerSettings, old_graph: &'a Graph) -> Self {
        Optimizer {
            settings,
            hidden_values: find_single_use_values(old_graph),
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
        if let Some(result) = self.try_fuse_clamp(old_start) { return Some(result); }
        if let Some(result) = self.try_fuse_conv_affine(old_start) { return Some(result); }

        None
    }

    fn try_fuse_clamp(&mut self, old_start: Value) -> Option<Value> {
        let mut total_min = f32::NEG_INFINITY;
        let mut total_max = f32::INFINITY;

        let old_input = self.follow_if(old_start, |_, _, operation| {
            if let &Operation::Clamp { input: old_input, min, max } = operation {
                total_min = f32::max(total_min, min);
                total_max = f32::min(total_max, max);
                Some(old_input)
            } else {
                None
            }
        })?;

        let new_input = self.map(old_input);
        let new_output = self.new_graph.clamp(new_input, total_min, total_max);
        Some(new_output)
    }

    fn try_fuse_conv_affine(&mut self, old_start: Value) -> Option<Value> {
        let group = self.try_build_affine_group(old_start)?;

        let new_input = self.map(group.old_input());
        let new_start = group.apply_fused(self.settings, &mut self.new_graph, new_input);

        Some(new_start)
    }

    pub fn follow_const(&self, start: Value) -> Option<&[f32]> {
        let input = self.follow_views(start);

        if let Operation::Constant { data } = &self.old_graph[input].operation {
            Some(data.as_slice())
        } else {
            None
        }
    }

    pub fn follow_views(&self, start: Value) -> Value {
        self.follow_if(start, |_, _, operation| {
            if let &Operation::View { input } = operation {
                Some(input)
            } else {
                None
            }
        }).unwrap_or(start)
    }

    pub fn follow_if(&self, start: Value, mut next: impl FnMut(&Graph, Value, &Operation) -> Option<Value>) -> Option<Value> {
        let mut curr = start;

        loop {
            if !self.hidden_values.contains(&curr) {
                break;
            }

            if let Some(next) = next(&self.old_graph, curr, &self.old_graph[curr].operation) {
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

pub fn find_single_use_values(graph: &Graph) -> HashSet<Value> {
    let all_inputs = graph.values()
        .flat_map(|v| graph[v].operation.inputs())
        .collect_vec();

    graph.values()
        .filter(|&value| {
            let occurrences = all_inputs.iter()
                .filter(|&&other| other == value)
                .count();
            occurrences < 2
        })
        .collect()
}
