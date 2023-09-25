use crate::graph::Graph;
use crate::optimizer::core::Optimizer;

mod affine;
mod core;
pub mod recurse;

/// Settings for the optimizer.
///
/// Use `Default::default()` to get reasonable defaults.
#[derive(Debug, Copy, Clone)]
pub struct OptimizerSettings {
    /// If `false`, don't do any optimization at all.
    pub optimize: bool,
    /// If `true`, convert a bias operation followed by a convolution _through_ the convolution,
    /// even in cases where this requires switching to a non-spatially-broadcasted bias constant.
    pub force_bias_through_conv: bool,
    /// If `true`, try fusing the right sequence of operations into a single LayerNorm operation.
    pub fuse_layernorm: bool,
    /// If `true`, convert a division by a constant into multiplication by the inverse consent.
    pub div_to_mul: bool,
}

/// Optimize the given graph according to the given settings. Returns a new, fully independent graph.
pub fn optimize_graph(graph: &Graph, settings: OptimizerSettings) -> Graph {
    if !settings.optimize {
        return graph.clone();
    }

    let mut optimizer = Optimizer::new(settings, graph);

    // ensure all inputs are copied over in the same order
    for &old_input in graph.inputs() {
        let shape = graph[old_input].shape.clone();
        let new_input = optimizer.new_graph.input(shape);
        optimizer.insert_mapping(old_input, new_input);
    }

    // register all outputs, again in the same order as before
    for &old_output in graph.outputs() {
        let new_output = optimizer.visit_completely(old_output);
        optimizer.new_graph.output(new_output);
    }

    optimizer.new_graph
}

#[allow(clippy::derivable_impls)]
impl Default for OptimizerSettings {
    fn default() -> Self {
        OptimizerSettings {
            optimize: true,
            force_bias_through_conv: false,
            fuse_layernorm: true,
            div_to_mul: true,
        }
    }
}
