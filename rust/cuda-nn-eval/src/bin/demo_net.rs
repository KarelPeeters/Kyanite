use npyz::npz::NpzArchive;

use cuda_sys::wrapper::handle::{cuda_device_count, CudaStream, CudnnHandle, Device};
use cuda_nn_eval::tower_net::TowerShape;
use cuda_nn_eval::load::load_params_from_npz;
use cuda_nn_eval::executor::CudaGraphExecutor;

fn main() {
    main_thread()
}

fn main_thread() {
    println!("Cuda device count: {}", cuda_device_count());
    let stream = CudaStream::new(Device::new(0));
    let handle = CudnnHandle::new(stream);

    let shape = TowerShape {
        board_size: 7,
        input_channels: 3,
        policy_channels: 17,

        tower_channels: 32,
        tower_depth: 2,
        wdl_hidden_size: 16,
    };

    let batch_size = 1;
    let graph = shape.to_graph(batch_size);

    println!("{:?}", graph);

    let mut npz = NpzArchive::open("../data/derp/basic_res_model/params.npz").unwrap();
    let params = load_params_from_npz(&graph, &mut npz, handle.device());

    // println!("{:?}", params);

    // let fused = FusedGraph::new(&graph);
    // println!("{:?}", fused);

    let mut executor = CudaGraphExecutor::new(handle, &graph, params);
    // println!("{:?}", executor);

    let batch_size = batch_size as usize;
    let input = vec![0.0; batch_size * 3 * 7 * 7];
    let mut output_wdl = vec![0.0; batch_size * 3];
    let mut output_policy = vec![0.0; batch_size * 17 * 7 * 7];

    executor.eval(&[&input], &mut [&mut output_wdl, &mut output_policy]);

    println!("{:?}", &output_wdl);
}
