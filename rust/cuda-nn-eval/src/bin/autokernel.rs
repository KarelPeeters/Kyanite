extern crate core;

use itertools::Itertools;

use cuda_nn_eval::device_tensor::DeviceTensor;
use cuda_nn_eval::shape::StridedShape;
use cuda_sys::bindings::{cudaError, cudaStream_t, cudnnOpTensorOp_t};
use cuda_sys::wrapper::descriptor::TensorOpDescriptor;
use cuda_sys::wrapper::group::TensorOpArgs;
use cuda_sys::wrapper::handle::{CudaStream, CudnnHandle, Device};
use cuda_sys::wrapper::rtc::args::KernelArgs;
use cuda_sys::wrapper::rtc::core::{CuFunction, CuModule, Dim3};
use cuda_sys::wrapper::status::Status;

fn main() {
    unsafe { main_inner() }
}

unsafe fn push_tensor(args: &mut KernelArgs, shape: &[usize], tensor: &DeviceTensor) {
    assert_eq!(tensor.shape().shape(), shape);
    assert_eq!(tensor.shape().rank(), 2);

    args.push_int(tensor.shape().strides()[0] as i32);
    args.push_int(tensor.shape().strides()[1] as i32);
    args.push(tensor.ptr().ptr());
}

unsafe fn launch(stream: &CudaStream, func: &CuFunction, a: &DeviceTensor, b: &DeviceTensor, c: &DeviceTensor) {
    let shape = a.shape().shape();
    assert_eq!(shape.len(), 2);

    let args = {
        let mut args = KernelArgs::new();

        args.push_int(shape[0] as i32);
        args.push_int(shape[1] as i32);

        push_tensor(&mut args, shape, a);
        push_tensor(&mut args, shape, b);
        push_tensor(&mut args, shape, c);

        args.finish()
    };

    let blocks = 128;
    let threads_per_block = 128;

    // let threads = blocks * threads_per_block;
    // let items_per_thread = (a.shape().size() as u32 + threads - 1) / threads;
    // println!("{} items per thread", items_per_thread);

    func.launch_kernel(Dim3::single(blocks), Dim3::single(threads_per_block), 0, stream, &args);
}

unsafe fn main_inner() {
    let template_path = "cuda-nn-eval/cuda/templates/autokernel.cu";
    let template = std::fs::read_to_string(template_path).unwrap();

    let device = Device::new(0);
    let handle = CudnnHandle::new(device);
    let stream = handle.stream();

    let module = CuModule::from_source(&template, Some(template_path), device);
    println!("{}", module.log);

    let module = module.module.unwrap();
    let func = module.get_function("foo_kernel").unwrap();

    let a_inner = DeviceTensor::alloc_simple(device, vec![1024, 256 * 8 * 8]);
    let b_inner = DeviceTensor::alloc_simple(device, vec![1, 1]);
    let c_inner = DeviceTensor::alloc_simple(device, vec![1024, 256 * 8 * 8]);

    let a_large = a_inner.clone();
    let b_large = b_inner.repeat_unary(0, 1024).repeat_unary(1, 256 * 8 * 8);
    let c_large = c_inner.clone();

    let iterations = 100;

    let start = stream.record_new_event();
    for _ in 0..iterations {
        launch(stream, &func, &a_large, &b_large, &c_large);
        // launch_foo(stream, &a_large, &b_large);
    }
    let end = stream.record_new_event();
    stream.synchronize();

    let time_per_kernel = end.time_elapsed_since(&start) / iterations as f32;
    let throughput = (4 * a_large.shape().size()) as f32 / time_per_kernel;

    println!("Delta per kernel: {} ms", time_per_kernel * 1000.0);
    println!("Kernels per second: {}", 1.0 / time_per_kernel);
    println!(
        "Manual throughput:\n  {} items/s\n  {} GBps",
        throughput,
        throughput / 1024f32.powi(3)
    );
}
