use bytemuck::{cast_slice, cast_slice_mut};

use cuda_sys::bindings::{cudnnDataType_t, cudnnTensorFormat_t, cudnnActivationMode_t};
use cuda_sys::wrapper::descriptor::{ConvolutionDescriptor, FilterDescriptor, TensorDescriptor, ActivationDescriptor};
use cuda_sys::wrapper::handle::CudnnHandle;
use cuda_sys::wrapper::mem::DeviceMem;
use cuda_sys::wrapper::operation::{run_conv, run_add_tensor, find_conv_algorithms, run_activation_in_place};

pub fn main() {
    let batch_size = 1000;

    let mut handle = CudnnHandle::new(0);
    let input_desc = TensorDescriptor::new(
        batch_size, 3, 7, 7,
        cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
    );
    let output_desc = TensorDescriptor::new(
        batch_size, 32, 7, 7,
        cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
    );
    let filter_desc = FilterDescriptor::new(
        32, 3, 3, 3,
        cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
    );

    let conv_desc = ConvolutionDescriptor::new(
        1, 1, 1, 1, 1, 1,
        cudnnDataType_t::CUDNN_DATA_FLOAT,
    );

    let algo_info = find_conv_algorithms(&mut handle, &conv_desc, &filter_desc, &input_desc, &output_desc)[0];
    println!("{:?}", algo_info);

    println!("input size: {}", input_desc.size());
    println!("output size: {}", output_desc.size());
    println!("filter size: {}", filter_desc.size());
    println!("workspace size: {}", algo_info.memory);

    let mut input_mem = DeviceMem::alloc(input_desc.size());
    let mut output_mem = DeviceMem::alloc(output_desc.size());
    let mut filter_mem = DeviceMem::alloc(filter_desc.size());
    let mut work_mem = DeviceMem::alloc(algo_info.memory);

    input_mem.copy_from_host(cast_slice(&vec![1f32; input_desc.size() / 4]));
    filter_mem.copy_from_host(cast_slice(&vec![2f32; filter_mem.size() / 4]));
    output_mem.copy_from_host(cast_slice(&vec![3f32; output_mem.size() / 4]));

    run_conv(
        &mut handle,
        &conv_desc, algo_info.algo,
        &mut work_mem,
        &filter_desc, &filter_mem,
        &input_desc, &input_mem,
        &output_desc, &mut output_mem,
    );

    let bias_desc = TensorDescriptor::new(
        1, 32, 1, 1,
        cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
    );
    let mut bias_mem = DeviceMem::alloc(bias_desc.size());
    bias_mem.copy_from_host(cast_slice(&vec![-30.0f32; bias_desc.size() / 4]));

    run_add_tensor(
        &mut handle,
        &bias_desc,
        &bias_mem,
        &output_desc,
        &mut output_mem,
    );

    let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_RELU, 0.0);
    run_activation_in_place(
        &mut handle,
        &activation_desc,
        &output_desc,
        &mut output_mem,
    );

    let mut input_copy = vec![0f32; input_desc.size() / 4];
    input_mem.copy_to_host(cast_slice_mut(&mut input_copy));
    let mut filter_copy = vec![0f32; filter_desc.size() / 4];
    filter_mem.copy_to_host(cast_slice_mut(&mut filter_copy));
    let mut output_copy = vec![0f32; output_desc.size() / 4];
    output_mem.copy_to_host(cast_slice_mut(&mut output_copy));

    println!("{:?}", &input_copy);
    println!("{:?}", &filter_copy);
    println!("{:?}", &output_copy);
}