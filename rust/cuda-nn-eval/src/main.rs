use nn_graph::onnx::load_graph_from_onnx_path;
use nn_graph::optimizer::optimize_graph;

fn main() {
    let graph = load_graph_from_onnx_path("../data/newer_loop/test-diri2/ttt/training/gen_130/network.onnx");
    println!("{}", graph);

    let optimized_graph = optimize_graph(&graph);
    println!("{}", optimized_graph);
}