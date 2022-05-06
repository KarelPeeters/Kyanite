extern crate core;

use cuda_nn_eval::device_tensor::DeviceTensor;
use cuda_nn_eval::shape::StridedShape;
use cuda_sys::bindings::cudnnOpTensorOp_t;
use cuda_sys::wrapper::descriptor::TensorOpDescriptor;
use cuda_sys::wrapper::group::TensorOpArgs;
use cuda_sys::wrapper::handle::{CudaStream, CudnnHandle, Device};
use cuda_sys::wrapper::mem::device::DevicePtr;
use cuda_sys::wrapper::rtc::args::KernelArgs;
use cuda_sys::wrapper::rtc::core::{CuFunction, CuModule, Dim3};

fn main() {
    unsafe { main_inner() }
}

unsafe fn push_tensor(args: &mut KernelArgs, shape: &[usize], tensor: &DeviceTensor) {
    assert_eq!(tensor.shape().shape(), shape);

    for &s in tensor.shape().strides() {
        args.push_int(s as i32);
    }

    args.push(tensor.ptr().ptr());
}

unsafe fn launch(
    stream: &CudaStream,
    func: &CuFunction,
    rank: usize,
    a: &DeviceTensor,
    b: &DeviceTensor,
    c: &DeviceTensor,
) {
    let shape = a.shape().shape();
    assert_eq!(rank, shape.len());

    let args = {
        let mut args = KernelArgs::new();

        let dense_shape = StridedShape::new_simple(shape.to_vec());
        args.push(dense_shape.size() as i32);
        for &s in dense_shape.strides() {
            args.push_int(s as i32);
        }

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
    let template_path = "cuda-nn-eval/cuda/templates/autokernel.cu";
    let template = std::fs::read_to_string(template_path).unwrap();

    let device = Device::new(0);
    let handle = CudnnHandle::new(device);
    let stream = handle.stream();

    let shape = vec![1024 * 256 * 8 * 8];

    let rank = shape.len();
    let operation = "a + b";

    let func_name = format!("foo_kernel<{}>", rank);
    let source = template.replace("$OPERATION$", operation);

    let module = CuModule::from_source(device, &source, Some(template_path), &[&func_name]);
    println!("{}", module.log);
    println!("{:?}", module.lowered_names);

    let func_lowered_name = module.lowered_names.get(&func_name).unwrap();

    let module = module.module.unwrap();
    let func = module.get_function(func_lowered_name).unwrap();

    let a_inner = DeviceTensor::alloc_simple(device, shape.clone());
    let b_inner = DeviceTensor::alloc_simple(device, shape.clone());
    let c_inner = DeviceTensor::alloc_simple(device, shape.clone());

    let a_large = a_inner.clone();
    // let a_large = a_inner.repeat_unary(0, 1024).repeat_unary(1, 256 * 8 * 8);
    let b_large = b_inner.clone();
    // let b_large = b_inner.repeat_unary(0, 1024).repeat_unary(1, 256 * 8 * 8);
    let c_large = c_inner.clone();

    println!("Autokernel");
    let time_manual = profile_kernel(&stream, || launch(stream, &func, rank, &a_large, &b_large, &c_large));
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
