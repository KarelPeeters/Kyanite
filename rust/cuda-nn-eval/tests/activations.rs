use bytemuck::{cast_slice, cast_slice_mut};

use cuda_sys::bindings::{cudnnActivationMode_t, cudnnDataType_t, cudnnTensorFormat_t};
use cuda_sys::wrapper::descriptor::ActivationDescriptor;
use cuda_sys::wrapper::group::Tensor;
use cuda_sys::wrapper::handle::{CudaStream, CudnnHandle, Device};
use cuda_sys::wrapper::operation::run_activation_in_place;
use itertools::Itertools;
use std::convert::identity;

#[test]
fn test_identity() {
    compare_activation(
        ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY, 0.0),
        identity
    );
}

#[test]
fn test_relu() {
    compare_activation(
        ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_RELU, 0.0),
        |x| x.max(0.0)
    );
}

#[test]
fn test_relu6() {
    compare_activation(
        ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_CLIPPED_RELU, 6.0),
        |x| x.clamp(0.0, 6.0),
    );
}

fn compare_activation(activation: ActivationDescriptor, expected: impl Fn(f32) -> f32) {
    let input = vec![-1.0, 0.0, 1.0, 8.0];
    let expected = input.iter().copied().map(expected).collect_vec();
    let output = apply_activation(activation, &input);

    println!("input: {:?}", input);
    println!("expected: {:?}", expected);
    println!("output: {:?}", output);

    assert_eq!(expected, output);
}

fn apply_activation(activation: ActivationDescriptor, input: &[f32]) -> Vec<f32> {
    let device = Device::new(0);
    let mut handle = CudnnHandle::new(CudaStream::new(device));

    let mut tensor = Tensor::new(
        1, 1, 1, input.len() as i32,
        cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, device,
    );
    tensor.mem.copy_from_host(cast_slice(&input));

    run_activation_in_place(&mut handle, &activation, &tensor.desc, &mut tensor.mem);

    let mut output = vec![0.0; input.len()];
    tensor.mem.copy_to_host(cast_slice_mut(&mut output));

    output
}
