use bytemuck::{cast_slice, cast_slice_mut};

use cuda_sys::bindings::{cudnnActivationMode_t, cudnnOpTensorOp_t};
use cuda_sys::wrapper::descriptor::{ActivationDescriptor, TensorDescriptor, TensorOpDescriptor};
use cuda_sys::wrapper::handle::{CudnnHandle, Device};
use cuda_sys::wrapper::mem::device::DeviceMem;
use cuda_sys::wrapper::operation::{run_activation, run_tensor_op};

unsafe fn test_restride(f: impl FnOnce(&CudnnHandle, &TensorDescriptor, &DeviceMem, &TensorDescriptor, &DeviceMem)) {
    let device = Device::new(0);
    let handle = CudnnHandle::new(device);

    let input = DeviceMem::alloc(6 * 4, device);
    let input_data: Vec<f32> = vec![-1.0, 0.0, 2.0, 0.0, -3.0, 0.0];
    let input_desc = TensorDescriptor::new(vec![3, 1, 1, 1], vec![2, 1, 1, 1]);

    let output = DeviceMem::alloc(3 * 4, device);
    let mut output_data: Vec<f32> = vec![0.0; 3];
    let output_desc = TensorDescriptor::new(vec![3, 1, 1, 1], vec![1, 1, 1, 1]);

    input.copy_from_host(cast_slice(&input_data));
    f(&handle, &input_desc, &input, &output_desc, &output);
    output.copy_to_host(cast_slice_mut(&mut output_data));

    assert_eq!(output_data, vec![-1.0, 2.0, -3.0]);
}

/// ignored because this does not actually work, it was was just an idea
#[test]
#[ignore]
fn restride_with_activation() {
    unsafe {
        test_restride(|handle, input_desc, input, output_desc, output| {
            let act_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY, 0.0);
            run_activation(handle, &act_desc, input_desc, input, output_desc, output)
        })
    }
}

#[test]
fn restride_with_bias() {
    unsafe {
        test_restride(|handle, input_desc, input, output_desc, output| {
            let op_desc = TensorOpDescriptor::new(cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD);

            let zero = DeviceMem::alloc(4, handle.device());
            zero.copy_from_host(cast_slice(&[0f32]));
            let zero_desc = TensorDescriptor::new(vec![1, 1, 1, 1], vec![1, 1, 1, 1]);

            run_tensor_op(
                handle,
                &op_desc,
                1.0,
                input_desc,
                input,
                0.0,
                &zero_desc,
                &zero,
                0.0,
                output_desc,
                output,
            )
        })
    }
}