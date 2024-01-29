use bytemuck::{cast_slice, cast_slice_mut};
use itertools::Itertools;

use kn_cuda_sys::bindings::{cudnnDataType_t, cudnnOpTensorOp_t};
use kn_cuda_sys::wrapper::descriptor::{TensorDescriptor, TensorOpDescriptor};
use kn_cuda_sys::wrapper::handle::{CudaDevice, CudnnHandle};
use kn_cuda_sys::wrapper::operation::run_tensor_op;

#[test]
fn test_negative_stride() {
    let device = CudaDevice::new(0).unwrap();
    let handle = CudnnHandle::new(device);

    let input_desc = TensorDescriptor::new(vec![1, 1, 1, 8], vec![1, 1, 1, -1], cudnnDataType_t::CUDNN_DATA_FLOAT);
    let output_desc = TensorDescriptor::new(vec![1, 1, 1, 8], vec![1, 1, 1, 1], cudnnDataType_t::CUDNN_DATA_FLOAT);

    let input_data = (0..8).map(|x| x as f32).collect_vec();
    let mut output_data = vec![0f32; 8];

    let input = device.alloc(input_desc.size_bytes());
    let output = device.alloc(input_desc.size_bytes());

    let op_desc = TensorOpDescriptor::new(
        cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD,
        cudnnDataType_t::CUDNN_DATA_FLOAT,
    );

    let zero = device.alloc(4);
    let zero_desc = TensorDescriptor::new(vec![1, 1, 1, 1], vec![1, 1, 1, 1], cudnnDataType_t::CUDNN_DATA_FLOAT);

    unsafe {
        input.copy_linear_from_host(cast_slice(&input_data));
        zero.copy_linear_from_host(cast_slice(&[0f32]));

        handle.stream().synchronize();

        run_tensor_op(
            &handle,
            &op_desc,
            1.0,
            &input_desc,
            &input.offset_bytes(4 * 8 - 4),
            0.0,
            &zero_desc,
            &zero,
            0.0,
            &output_desc,
            &output,
        );

        handle.stream().synchronize();

        output.copy_linear_to_host(cast_slice_mut(&mut output_data));
    }

    println!("{:?}", input_data);
    println!("{:?}", output_data);
}
