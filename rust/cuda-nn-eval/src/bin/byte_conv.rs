use std::time::Instant;

use cuda_sys::bindings::{cudnnConvolutionForward, cudnnConvolutionFwdAlgo_t, cudnnDataType_t, cudnnTensorFormat_t};
use cuda_sys::wrapper::descriptor::{ConvolutionDescriptor, FilterDescriptor, TensorDescriptor};
use cuda_sys::wrapper::handle::{CudnnHandle, Device};
use cuda_sys::wrapper::mem::device::DeviceMem;
use cuda_sys::wrapper::status::Status;

// baseline: 100k evals/s
// NHWC: could not get working, maybe because of the algo?
// different algos, only either 10%, 25% or 40% of throughput
// type HALF: 60%
// type BFLOAT16: CUDNN_STATUS_ARCH_MISMATCH
// type INT8:

unsafe fn main_inner() {
    let device = Device::new(0);
    let handle = CudnnHandle::new(device);

    //TODO also try different strides for input/output and filter
    let data_type = cudnnDataType_t::CUDNN_DATA_INT8;
    let weight_format = cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
    let algo = cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM;

    let b: i32 = 1024;
    let c: i32 = 32;
    let s = 8;

    println!("Using data type {:?}", data_type);
    println!("IO shape: {}x{}x{}x{}", b, c, s, s);

    let io_desc = TensorDescriptor::new_with_type(vec![b, c, s, s], vec![c * s * s, s * s, s, 1], data_type);
    let f_desc = FilterDescriptor::new_with_type_format(c, c, 3, 3, data_type, weight_format);
    let conv_desc = ConvolutionDescriptor::new(1, 1, 1, 1, 1, 1);

    println!("IO size: {}", io_desc.size_bytes());
    let workspace_size = conv_desc.workspace_size(&handle, algo, &io_desc, &f_desc, &io_desc);
    println!("Workspace size: {}", workspace_size);

    let x_mem = DeviceMem::alloc(io_desc.size_bytes(), device);
    let y_mem = DeviceMem::alloc(io_desc.size_bytes(), device);
    let f_mem = DeviceMem::alloc(f_desc.size_bytes(), device);
    let w_mem = DeviceMem::alloc(workspace_size, device);

    let run = || {
        cudnnConvolutionForward(
            handle.inner(),
            &1f32 as *const _ as *const _,
            io_desc.inner(),
            x_mem.ptr(),
            f_desc.inner(),
            f_mem.ptr(),
            conv_desc.inner(),
            algo,
            w_mem.ptr(),
            workspace_size,
            &0f32 as *const _ as *const _,
            io_desc.inner(),
            y_mem.ptr(),
        )
        .unwrap();
    };

    for _ in 0..10 {
        run()
    }

    let start = Instant::now();
    let iterations = 10_000;
    for _ in 0..iterations {
        run()
    }

    let delta = start.elapsed();
    let throughput = (b * iterations) as f32 / delta.as_secs_f32();
    println!("Took: {:?}", delta);
    println!("Throughput: {} convs/s", throughput);
    println!("Throughput: {} evals/s", throughput / 20.0);
}

fn main() {
    unsafe { main_inner() }
}
