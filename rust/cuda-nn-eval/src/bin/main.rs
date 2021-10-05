use std::time::Instant;

use cuda_nn_eval::executor::CudnnExecutor;
use cuda_sys::wrapper::handle::{CudaStream, CudnnHandle, Device};
use nn_graph::onnx::load_onnx_graph;

fn main() {
    let graph = load_onnx_graph("ignored/network_6168_old.onnx");
    println!("{:?}", graph);

    let batch_size = 100;

    let handle = CudnnHandle::new(CudaStream::new(Device::new(0)));
    let mut executor = CudnnExecutor::new(handle, &graph, batch_size);

    let input = vec![0.0; batch_size * 21 * 8 * 8];

    loop {
        let start = Instant::now();

        let outputs = executor.evaluate(&[&input]);

        let delta = (Instant::now() - start).as_secs_f32();
        let thoughput = batch_size as f32 / delta;

        println!("Throughput: {:.2}", thoughput);
    }
}
