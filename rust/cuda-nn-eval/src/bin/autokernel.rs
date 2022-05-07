extern crate core;

use cuda_nn_eval::autokernel::scalar::ScalarKernel;
use cuda_nn_eval::device_tensor::DeviceTensor;
use cuda_sys::bindings::cudnnOpTensorOp_t;
use cuda_sys::wrapper::descriptor::TensorOpDescriptor;
use cuda_sys::wrapper::group::TensorOpArgs;
use cuda_sys::wrapper::handle::{CudaStream, CudnnHandle, Device};
use cuda_sys::wrapper::mem::device::DevicePtr;
use itertools::Itertools;

fn main() {
    unsafe { main_inner() }
}

unsafe fn profile_kernel(stream: &CudaStream, f: impl Fn()) -> f32 {
    let iterations = 100;

    let start = stream.record_new_event();
    for _ in 0..iterations {
        f();
    }
    let end = stream.record_new_event();
    stream.synchronize();

    let time_per_kernel = end.time_elapsed_since(&start) / iterations as f32;
    time_per_kernel
}

unsafe fn main_inner() {
    let device = Device::new(0);
    let handle = CudnnHandle::new(device);
    let stream = handle.stream();

    println!("Building buffers");
    let batch_size: usize = 1024;
    let inner_shape = vec![256, 8, 8];
    let full_shape = [&[batch_size], &*inner_shape].concat();
    let full_size: usize = full_shape.iter().copied().product();

    let a_inner = DeviceTensor::alloc_simple(device, full_shape.clone());
    let b_inner = DeviceTensor::alloc_simple(device, full_shape.clone());
    let c_inner = DeviceTensor::alloc_simple(device, full_shape.clone());

    let a_large = a_inner.clone();
    // let a_large = a_inner.repeat_unary(0, 1024).repeat_unary(1, 256 * 8 * 8);
    let b_large = b_inner.clone();
    // let b_large = b_inner.repeat_unary(0, 1024).repeat_unary(1, 256 * 8 * 8);
    let c_large = c_inner.clone();

    let operands = vec![a_large, b_large, c_large.clone()];

    let kernel = ScalarKernel::new(
        device.compute_capability(),
        "*x0 = *x1 + *x2;",
        inner_shape,
        vec![String::from("float"); operands.len()],
        operands.iter().map(|op| op.shape().strides().to_vec()).collect_vec(),
    );

    let time_manual = profile_kernel(&stream, || kernel.run(&stream, &operands));
    let tp_manual = (4 * full_size) as f32 / time_manual;

    println!("  Delta per kernel: {} ms", time_manual * 1000.0);
    println!("  Kernels per second: {}", 1.0 / time_manual);
    println!(
        "  Item throughput:\n  {} items/s\n  {} GBps",
        tp_manual,
        tp_manual / 1024f32.powi(3)
    );

    // use inner tensors and hope that broadcasting works out
    let args = TensorOpArgs::<DevicePtr> {
        op_desc: TensorOpDescriptor::new(cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD),
        alpha_1: 1.0,
        input_1_desc: a_inner.shape().descriptor(),
        input_1_ptr: a_inner.ptr().clone(),
        alpha_2: 1.0,
        input_2_desc: b_inner.shape().descriptor(),
        input_2_ptr: b_inner.ptr().clone(),
        beta: 0.0,
        output_desc: c_large.shape().descriptor(),
        output_ptr: c_large.ptr().clone(),
    };

    println!("Cudnn TensorOp");
    let time_cudnn = profile_kernel(stream, || args.run(&handle));
    let tp_cudnn = (4 * full_size) as f32 / time_cudnn;
    println!("  Delta per kernel: {} ms", time_cudnn * 1000.0);
    println!("  Kernels per second: {}", 1.0 / time_cudnn);
    println!(
        "  Item throughput:\n  {} items/s\n  {} GBps",
        tp_cudnn,
        tp_cudnn / 1024f32.powi(3)
    );
}
