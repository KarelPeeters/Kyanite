use cuda_nn_eval::executor::CudnnExecutor;
use cuda_sys::wrapper::handle::Device;
use nn_graph::graph::Graph;
use nn_graph::shape;
use nn_graph::shape::Size;

fn main() {
    for io_size in [8, 9, 10] {
        println!("{}", profile_conv(128, 128, 128, io_size, 3, true));
    }
}

fn profile_conv(batch_size: usize, input_channels: usize, output_channels: usize, io_size: usize, kernel_size: usize, use_graph: bool) -> f32 {
    let input_shape = shape![Size::BATCH, input_channels, io_size, io_size];
    let kernel_shape = shape![output_channels, input_channels, kernel_size, kernel_size];

    let mut graph = Graph::new();
    let input = graph.input(input_shape.clone());
    let kernel_size = kernel_shape.size().unwrap_fixed("");
    let filter = graph.constant(kernel_shape, vec![2.0; kernel_size]);
    let output = graph.conv(input, filter, 0, 0);
    graph.output(output);

    let input = vec![2.0; input_shape.size().eval(batch_size)];

    let device = Device::new(0);
    let mut exec = CudnnExecutor::new(device, &graph, batch_size, use_graph);

    let samples = 1000;

    // warmup
    for _ in 0..samples {
        exec.evaluate(&[&input]);
    }

    //actual profiling
    exec.set_profile(true);
    let total = (0..samples).map(|_| {
        exec.evaluate(&[&input]);
        exec.last_profile().unwrap().conv
    }).sum::<f32>();

    total / samples as f32
}