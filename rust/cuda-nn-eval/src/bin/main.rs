use cuda_nn_eval::executor::CudaGraphExecutor;
use cuda_nn_eval::graph::Graph;
use cuda_nn_eval::onnx::load_onnx_graph;
use cuda_sys::wrapper::handle::{CudaStream, CudnnHandle, Device};
use rand::{thread_rng, Rng};

fn own_impl(path: &str, batch_size: usize, input: &[f32], output_wdl: &mut [f32], output_policy: &mut [f32]) {
    let graph = load_onnx_graph(path, batch_size as i32, Device::new(0));
    println!("{:?}", graph);

    let handle = CudnnHandle::new(CudaStream::new(Device::new(0)));
    let mut executor = CudaGraphExecutor::new(handle, &graph);

    let input = vec![0.0; 100 * 3 * 7 * 7];
    let mut output_wdl = vec![0.0; 100 * 3];
    let mut output_policy = vec![0.0; 100 * 17 * 7 * 7];

    executor.run(&[&input], &mut [&mut output_wdl, &mut output_policy]);
}

fn main() {
    let path = "../data/derp/basic_res_model/model.onnx";
    let batch_size = 100;

    let mut rng = thread_rng();
    let input: Vec<f32> = (0..(batch_size * 3 * 7 * 7)).map(|_| rng.gen()).collect();



    own_impl()
}