use cuda_nn_eval::cpu_executor::CpuExecutor;
use cuda_nn_eval::graph::Graph;

fn main() {
    let n = 1;

    let graph = {
        let mut graph = Graph::empty();

        let input = graph.input([n as i32, 2, 2, 2]);
        let filter = graph.constant([1, 2, 1, 1], vec![1.0, -1.0]);
        let bias = graph.constant([1, 1, 1, 1], vec![0.0]);
        let conv_output = graph.conv(input, filter, 0);
        let bias_output = graph.bias(conv_output, bias);
        graph.output(bias_output);

        graph
    };

    let mut executor = CpuExecutor::new(&graph);

    let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let mut output = vec![0.0; 4 * n];
    executor.evaluate(&[&input], &mut [&mut output]);

    println!("{:?}", input);
    println!("{:?}", output);
}