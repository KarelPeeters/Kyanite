use bytemuck::{cast_slice, cast_slice_mut};
use itertools::Itertools;

use cuda_nn_eval::{Device, kernels};
use cuda_sys::wrapper::handle::CudaStream;
use cuda_sys::wrapper::mem::device::DeviceMem;
use cuda_sys::wrapper::status::Status;

fn main() {
    let device = Device::new(0);
    let stream = CudaStream::new(device);
    let input = DeviceMem::alloc(128 * 4, device);
    let output = DeviceMem::alloc(128 * 4, device);

    let input_data = (0..128).map(|x| 0.0 + x as f32).collect_vec();
    let mut output_data = vec![0f32; 128];

    let input_strides: Vec<i32> = vec![8, 1];
    let output_strides: Vec<i32> = vec![8, 1];
    let dense_strides: Vec<i32> = vec![8, 1];

    unsafe {
        input.copy_from_host(cast_slice(&input_data));

        kernels::stridedFloatCopy(
            stream.inner(),
            2,
            64,
            input_strides.as_ptr(), output_strides.as_ptr(), dense_strides.as_ptr(),
            input.ptr() as *const f32, output.ptr() as *mut f32,
        ).unwrap();

        output.copy_to_host(cast_slice_mut(&mut output_data));
    }

    println!("{:?}", output_data);

    /*    let graph = load_graph_from_onnx_path("../data/newer_loop/test-diri2/ttt/training/gen_130/network.onnx");
        println!("{}", graph);

        let optimized_graph = optimize_graph(&graph);
        println!("{}", optimized_graph);*/
}