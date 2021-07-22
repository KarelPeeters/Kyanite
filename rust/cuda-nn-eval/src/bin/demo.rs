use bytemuck::{cast_slice, cast_slice_mut};

use cuda_sys::bindings::{cudnnConvolutionForward, cudnnDataType_t, cudnnGetConvolutionForwardWorkspaceSize, cudnnTensorFormat_t};
use cuda_sys::wrapper::descriptor::{ConvolutionDescriptor, FilterDescriptor, TensorDescriptor};
use cuda_sys::wrapper::handle::CudnnHandle;
use cuda_sys::wrapper::mem::DeviceMem;
use cuda_sys::wrapper::status::Status;

pub fn main() {
    let batch_size = 1;

    let mut handle = CudnnHandle::new();
    let input_desc = TensorDescriptor::new(
        batch_size, 1, 7, 7,
        cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
    );
    let output_desc = TensorDescriptor::new(
        batch_size, 1, 7, 7,
        cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
    );
    let filter_desc = FilterDescriptor::new(
        1, 1, 3, 3,
        cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
    );

    let conv_desc = ConvolutionDescriptor::new(
        1, 1, 1, 1, 1, 1,
        cudnnDataType_t::CUDNN_DATA_FLOAT,
    );

    let algo_info = conv_desc.find_algorithms(&mut handle, &input_desc, &filter_desc, &output_desc)[0];

    println!("{:?}", algo_info);

    let workspace_size = unsafe {
        let mut workspace_size: usize = 0;
        cudnnGetConvolutionForwardWorkspaceSize(
            handle.inner(),
            input_desc.inner(),
            filter_desc.inner(),
            conv_desc.inner(),
            output_desc.inner(),
            algo_info.algo,
            &mut workspace_size as *mut _,
        ).unwrap();
        workspace_size
    };

    println!("input size: {}", input_desc.size());
    println!("output size: {}", output_desc.size());
    println!("filter size: {}", filter_desc.size());
    println!("workspace size (auto): {}", algo_info.memory);
    println!("workspace size (direct): {}", workspace_size);

    let mut input_mem = DeviceMem::alloc(input_desc.size());
    let mut output_mem = DeviceMem::alloc(output_desc.size());
    let mut filter_mem = DeviceMem::alloc(filter_desc.size());
    let workspace_mem = DeviceMem::alloc(algo_info.memory);

    input_mem.copy_from_host(cast_slice(&vec![1f32; input_desc.size() / 4]));
    filter_mem.copy_from_host(cast_slice(&vec![2f32; filter_mem.size() / 4]));
    output_mem.copy_from_host(cast_slice(&vec![3f32; output_mem.size() / 4]));

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe {
        cudnnConvolutionForward(
            handle.inner(),
            &alpha as *const _ as *const _,
            input_desc.inner(),
            input_mem.inner(),
            filter_desc.inner(),
            filter_mem.inner(),
            conv_desc.inner(),
            algo_info.algo,
            workspace_mem.inner(),
            workspace_mem.size(),
            &beta as *const _ as *const _,
            output_desc.inner(),
            output_mem.inner(),
        ).unwrap();
    }

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