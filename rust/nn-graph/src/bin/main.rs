use nn_graph::onnx::load_onnx_graph;

fn main() {
    let graph = load_onnx_graph("ignored/network_6168_old.onnx");
    println!("{:?}", graph);
}
