use kn_cuda_eval::executor::CudaExecutor;
use kn_cuda_sys::wrapper::handle::Device;
use kn_graph::dtype::{DTensor, DType, Tensor};
use kn_graph::graph::Graph;
use kn_graph::shape;
use kn_graph::shape::Size;

fn main() {
    for io_size in [8, 9, 10] {
        println!("{}", profile_conv(128, 128, 128, io_size, 3));
    }
}

fn profile_conv(
    batch_size: usize,
    input_channels: usize,
    output_channels: usize,
    io_size: usize,
    kernel_size: usize,
) -> f32 {
    let input_shape = shape![Size::BATCH, input_channels, io_size, io_size];
    let kernel_shape = shape![output_channels, input_channels, kernel_size, kernel_size];

    let mut graph = Graph::new();
    let shape = input_shape.clone();
    let input = graph.input(shape, DType::F32);
    let kernel_size = kernel_shape.size().unwrap_fixed("");
    let filter = graph.constant::<f32>(kernel_shape, vec![2.0; kernel_size]);
    let output = graph.conv(input, filter, 1, 1, 0, 0);
    graph.output(output);

    let input = DTensor::F32(Tensor::from_elem(input_shape.eval(batch_size).dims, 2.0));
    let inputs = &[input];

    let device = Device::new(0);
    let mut exec = CudaExecutor::new(device, &graph, batch_size);

    let samples = 1000;

    // warmup
    for _ in 0..samples {
        exec.evaluate(inputs);
    }

    //actual profiling
    exec.set_profile(true);
    let total = (0..samples)
        .map(|_| {
            exec.evaluate(inputs);
            exec.last_profile().unwrap().conv
        })
        .sum::<f32>();

    total / samples as f32
}
