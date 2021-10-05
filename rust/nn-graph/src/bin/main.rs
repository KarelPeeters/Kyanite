use ndarray::IxDyn;

use nn_graph::cpu::{cpu_execute_graph, Tensor};
use nn_graph::onnx::load_onnx_graph;

fn main() {
    let graph = load_onnx_graph("ignored/network_6168_old.onnx");
    println!("{:?}", graph);

    let batch_size = 100;
    let input = Tensor::ones(IxDyn(&[batch_size, 21, 8, 8]));

    let result = cpu_execute_graph(&graph, batch_size, &[&input]);
    println!("{:?}", result);
}
