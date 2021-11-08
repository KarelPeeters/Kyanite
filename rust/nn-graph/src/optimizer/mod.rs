use crate::graph::Graph;
pub use crate::optimizer::core::find_single_use_values;
use crate::optimizer::core::Optimizer;

mod core;
mod affine;

#[derive(Debug, Copy, Clone)]
pub struct OptimizerSettings {
    pub force_bias_through_conv: bool,
}

pub fn optimize_graph(graph: &Graph, settings: OptimizerSettings) -> Graph {
    let mut optimizer = Optimizer::new(settings, graph);

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

impl Default for OptimizerSettings {
    fn default() -> Self {
        OptimizerSettings { force_bias_through_conv: false }
    }
}