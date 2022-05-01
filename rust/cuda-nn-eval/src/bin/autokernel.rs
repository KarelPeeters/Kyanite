extern crate core;

use cuda_nn_eval::device_tensor::DeviceTensor;
use cuda_nn_eval::shape::StridedShape;
use cuda_sys::bindings::cudnnOpTensorOp_t;
use cuda_sys::wrapper::descriptor::TensorOpDescriptor;
use cuda_sys::wrapper::group::TensorOpArgs;
use cuda_sys::wrapper::handle::{CudaStream, CudnnHandle, Device};
use cuda_sys::wrapper::rtc::args::KernelArgs;
use cuda_sys::wrapper::rtc::core::{CuFunction, CuModule, Dim3};
use itertools::Itertools;

fn main() {
    unsafe { main_inner() }
}

unsafe fn launch(stream: &CudaStream, func: &CuFunction, a: &DeviceTensor, b: &DeviceTensor) {
    let args = {
        // int shape[2], int stride0[2], int stride1[2], float *x0, float *x1
        let mut args = KernelArgs::new();
        assert_eq!(a.shape().shape(), b.shape().shape());
        assert_eq!(a.shape().rank(), 2);
        args.push_int(a.shape().shape()[0] as i32);
        args.push_int(a.shape().shape()[1] as i32);
        args.push_int(a.shape().strides()[0] as i32);
        args.push_int(a.shape().strides()[1] as i32);
        args.push_int(b.shape().strides()[0] as i32);
        args.push_int(b.shape().strides()[1] as i32);
        args.push(a.ptr().ptr());
        args.push(b.ptr().ptr());
        args.finish()
    };

    func.launch_kernel(Dim3::single(128), Dim3::single(128), 0, stream, &args);
}

unsafe fn main_inner() {
    let template_path = "template.cu";
    let template = std::fs::read_to_string(template_path).unwrap();

    let device = Device::new(0);
    let handle = CudnnHandle::new(device);
    let stream = handle.stream();

    let module = CuModule::from_source(&template, Some(template_path), device);

    println!("{}", module.log);

    let zero = DeviceTensor::alloc_simple(device, vec![1, 1]);
    zero.copy_simple_from_host(&[0.0]);

    let a = DeviceTensor::alloc_simple(device, vec![8, 1]);
    let b = DeviceTensor::alloc_simple(device, vec![8, 3]);

    let buffer_a = (0..a.shape().size()).map(|i| i as f32).collect_vec();
    let mut buffer_b = vec![f32::NAN; b.shape().size()];

    let module = module.module.unwrap();
    let func = module.get_function("foo").unwrap();

    println!("a_ptr: {:?}", a.ptr().ptr());
    println!("b_ptr: {:?}", b.ptr().ptr());

    a.copy_from_host_staged(&buffer_a);

    launch(stream, &func, &a.repeat_unary(1, 3), &b);

    stream.synchronize();

    b.copy_to_host_staged(&mut buffer_b);

    println!("buffer_a: {:?}", buffer_a);
    println!("buffer_b: {:?}", buffer_b);

    println!("Start benchmarking");
    let a_inner = DeviceTensor::alloc_simple(device, vec![1, 1]);
    let b_inner = DeviceTensor::alloc_simple(device, vec![256, 256]);

    let a_large = a_inner.repeat_unary(0, 256).repeat_unary(1, 256);
    let b_large = b_inner;

    let iterations = 10;
    let start = stream.record_new_event();

    for _ in 0..iterations {
        launch(stream, &func, &a_large, &b_large);
    }

    let end = stream.record_new_event();
    stream.synchronize();

    let delta = end.time_elapsed_since(&start);
    let throughput = (a_large.shape().size() * iterations) as f32 / delta;
    println!(
        "Manual throughput:\n  {} items/s\n  {} GBps",
        throughput,
        throughput / 1024f32.powi(3)
    );
}
