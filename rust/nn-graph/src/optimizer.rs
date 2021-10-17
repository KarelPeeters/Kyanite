use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use crate::graph::{Graph, Operation, Value};

pub fn optimize_graph(graph: &Graph) -> Graph {
    let mut optimizer = Optimizer::new(graph);

    // ensure all inputs are copied over in the same order
    for &old_input in graph.inputs() {
        let shape = graph[old_input].shape.clone();
        let new_input = optimizer.new_graph.input(shape);
        optimizer.define(old_input, new_input);
    }

    // register all outputs, again in the same order as before
    for &old_output in graph.outputs() {
        let new_output = optimizer.map(old_output);
        optimizer.new_graph.output(new_output);
    }

    optimizer.new_graph
}

struct Optimizer<'a> {
    hidden_values: HashSet<Value>,
    old_graph: &'a Graph,
    new_graph: Graph,
    mapping: HashMap<Value, Value>,
}

impl<'a> Optimizer<'a> {
    fn new(old_graph: &'a Graph) -> Self {
        Optimizer {
            hidden_values: find_hidden_values(old_graph),
            new_graph: Graph::new(),
            old_graph,
            mapping: HashMap::default(),
        }
    }

    fn define(&mut self, old: Value, new: Value) {
        let prev = self.mapping.insert(old, new);
        assert!(prev.is_none());
    }

    fn map(&mut self, old_value: Value) -> Value {
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
        // clamp
        {
            let mut total_min = f32::NEG_INFINITY;
            let mut total_max = f32::INFINITY;

            if let Some(old_input) = self.follow_old(old_start, |_, _, operation| {
                if let &Operation::Clamp { input: old_input, min, max } = operation {
                    total_min = f32::max(total_min, min);
                    total_max = f32::min(total_max, max);
                    Some(old_input)
                } else {
                    None
                }
            }) {
                let new_input = self.map(old_input);
                let new_output = self.new_graph.clamp(new_input, total_min, total_max);
                return Some(new_output);
            }
        }

        // conv / bias
        {
            //TODO
        }

        None
    }

    fn follow_old(&self, start: Value, mut next: impl FnMut(&Graph, Value, &Operation) -> Option<Value>) -> Option<Value> {
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

fn find_hidden_values(graph: &Graph) -> HashSet<Value> {
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