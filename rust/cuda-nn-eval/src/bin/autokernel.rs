extern crate core;

use cuda_nn_eval::device_tensor::DeviceTensor;
use cuda_nn_eval::shape::StridedShape;
use cuda_sys::bindings::{cudaDeviceAttr, cudnnOpTensorOp_t};
use cuda_sys::wrapper::descriptor::TensorOpDescriptor;
use cuda_sys::wrapper::group::TensorOpArgs;
use cuda_sys::wrapper::handle::{CudaStream, CudnnHandle, Device};
use cuda_sys::wrapper::mem::device::DevicePtr;
use cuda_sys::wrapper::rtc::args::KernelArgs;
use cuda_sys::wrapper::rtc::core::{CuModule, Dim3};
use cuda_sys::wrapper::status::Status;
use itertools::Itertools;
use std::fmt::Write;

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
    let template_path = "cuda-nn-eval/cuda/autokernel/scalar.cu";
    let template = std::fs::read_to_string(template_path).unwrap();

    let device = Device::new(0);
    let handle = CudnnHandle::new(device);
    let stream = handle.stream();

    // operation settings
    let operation =
        "((float*) pointers[0])[offsets[0]] = ((float*) pointers[1])[offsets[1]] + ((float*) pointers[2])[offsets[2]];";
    let blocks = 128;
    let threads_per_block = 128;

    println!("Building buffers");
    let batch_size = 1024;
    let shape = vec![batch_size, 256, 8, 8];

    let a_inner = DeviceTensor::alloc_simple(device, shape.clone());
    let b_inner = DeviceTensor::alloc_simple(device, shape.clone());
    let c_inner = DeviceTensor::alloc_simple(device, shape.clone());

    let a_large = a_inner.clone();
    // let a_large = a_inner.repeat_unary(0, 1024).repeat_unary(1, 256 * 8 * 8);
    let b_large = b_inner.clone();
    // let b_large = b_inner.repeat_unary(0, 1024).repeat_unary(1, 256 * 8 * 8);
    let c_large = c_inner.clone();

    let operands = [&a_large, &b_large, &c_large];

    // map operands
    let operand_shapes = operands.iter().map(|op| op.shape()).collect_vec();
    let mut args = KernelArgs::new();
    args.push_int(batch_size as i32);
    for op in operands {
        args.push(op.ptr().ptr());
    }
    let args = args.finish();

    let replacements = build_replacements(operation, &operand_shapes);
    println!("Replacements: {:?}", replacements);

    let mut source = template;
    for (key, value) in replacements {
        assert!(source.contains(key), "Source does not contain '{}'", key);
        source = source.replace(key, &value);
    }

    println!("Source:\n{}\n\n", source);
    assert!(
        !source.contains("$"),
        "Leftover $-signs, probably because of a missing parameter argument?"
    );

    println!("Comping kernel");
    let kernel_name = "scalar_kernel";
    let module = CuModule::from_source(
        device.compute_capability(),
        &source,
        Some(template_path),
        &[kernel_name],
    );
    println!("{}", module.log);
    println!("{:?}", module.lowered_names);

    let func_lowered_name = module.lowered_names.get(kernel_name).unwrap();

    let module = module.module.unwrap();
    let func = module.get_function(func_lowered_name).unwrap();

    println!("Launching autokernel");

    let time_manual = profile_kernel(&stream, || {
        let grid_dim = Dim3::single(blocks);
        let block_dim = Dim3::single(threads_per_block);
        func.launch_kernel(grid_dim, block_dim, 0, &stream, &args).unwrap()
    });
    let tp_manual = (4 * a_large.shape().size()) as f32 / time_manual;

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
    let tp_cudnn = (4 * a_large.shape().size()) as f32 / time_cudnn;
    println!("  Delta per kernel: {} ms", time_cudnn * 1000.0);
    println!("  Kernels per second: {}", 1.0 / time_cudnn);
    println!(
        "  Item throughput:\n  {} items/s\n  {} GBps",
        tp_cudnn,
        tp_cudnn / 1024f32.powi(3)
    );
}
