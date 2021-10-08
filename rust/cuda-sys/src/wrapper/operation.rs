pub use crate::bindings::{cudnnActivationForward, cudnnAddTensor, cudnnConvolutionBiasActivationForward, cudnnConvolutionForward, cudnnConvolutionFwdAlgo_t, cudnnConvolutionFwdAlgoPerf_t, cudnnFindConvolutionForwardAlgorithm, cudnnGetConvolutionForwardAlgorithmMaxCount, cudnnStatus_t};
use crate::bindings::{cudnnOpTensor, cudnnPoolingForward};
use crate::wrapper::descriptor::{ActivationDescriptor, ConvolutionDescriptor, FilterDescriptor, PoolingDescriptor, TensorDescriptor, TensorOpDescriptor};
use crate::wrapper::handle::CudnnHandle;
use crate::wrapper::mem::DeviceMem;
use crate::wrapper::status::Status;

pub fn find_conv_algorithms(
    handle: &mut CudnnHandle,
    conv: &ConvolutionDescriptor,
    filter: &FilterDescriptor,
    input: &TensorDescriptor,
    output: &TensorDescriptor,
) -> Vec<cudnnConvolutionFwdAlgoPerf_t> {
    unsafe {
        let mut max_algo_count = 0;
        cudnnGetConvolutionForwardAlgorithmMaxCount(handle.inner(), &mut max_algo_count as *mut _).unwrap();

        let mut result = Vec::with_capacity(max_algo_count as usize);
        let mut algo_count = 0;
        cudnnFindConvolutionForwardAlgorithm(
            handle.inner(),
            input.inner(),
            filter.inner(),
            conv.inner(),
            output.inner(),
            max_algo_count,
            &mut algo_count as *mut _,
            result.as_mut_ptr(),
        ).unwrap();

        result.set_len(algo_count as usize);

        //remove non-supported algorithms and propagate errors
        result.retain(|a| {
            if a.status == cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED {
                false
            } else {
                a.status.unwrap();
                true
            }
        });

        result
    }
}

/// Run `output = conv(input, filter)`
pub unsafe fn run_conv(
    handle: &mut CudnnHandle,
    conv_desc: &ConvolutionDescriptor,
    algo: cudnnConvolutionFwdAlgo_t,
    work_mem: &DeviceMem,
    filter_desc: &FilterDescriptor,
    filter_mem: &DeviceMem,
    input_desc: &TensorDescriptor,
    input_mem: &DeviceMem,
    output_desc: &TensorDescriptor,
    output_mem: &DeviceMem,
) {
    assert_eq!(input_desc.size_bytes(), input_mem.len());
    assert_eq!(filter_desc.size_bytes(), filter_mem.len());
    assert_eq!(output_desc.size_bytes(), output_mem.len());

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    cudnnConvolutionForward(
        handle.inner(),
        &alpha as *const _ as *const _,
        input_desc.inner(),
        input_mem.ptr(),
        filter_desc.inner(),
        filter_mem.ptr(),
        conv_desc.inner(),
        algo,
        work_mem.ptr(),
        work_mem.len(),
        &beta as *const _ as *const _,
        output_desc.inner(),
        output_mem.ptr(),
    ).unwrap();
}

/// Run `output += input`. `input` can have dimensions of size 1 which are broadcasted to the shape of `output`.
pub unsafe fn run_add_tensor(
    handle: &mut CudnnHandle,
    input_desc: &TensorDescriptor,
    input_mem: &DeviceMem,
    output_desc: &TensorDescriptor,
    output_mem: &mut DeviceMem,
) {
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;

    cudnnAddTensor(
        handle.inner(),
        &alpha as *const _ as *const _,
        input_desc.inner(),
        input_mem.ptr(),
        &beta as *const _ as *const _,
        output_desc.inner(),
        output_mem.ptr(),
    ).unwrap();
}

/// Run `output = act(input)`.
pub unsafe fn run_activation(
    handle: &mut CudnnHandle,
    activation_desc: &ActivationDescriptor,
    input_desc: &TensorDescriptor,
    input_mem: &DeviceMem,
    output_desc: &TensorDescriptor,
    output_mem: &mut DeviceMem,
) {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    cudnnActivationForward(
        handle.inner(),
        activation_desc.inner(),
        &alpha as *const _ as *const _,
        input_desc.inner(),
        input_mem.ptr(),
        &beta as *const _ as *const _,
        output_desc.inner(),
        output_mem.ptr(),
    ).unwrap();
}

/// Runs `output = act(output)`.
pub unsafe fn run_activation_in_place(
    handle: &mut CudnnHandle,
    activation_desc: &ActivationDescriptor,
    data_desc: &TensorDescriptor,
    data_mem: &mut DeviceMem,
) {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    cudnnActivationForward(
        handle.inner(),
        activation_desc.inner(),
        &alpha as *const _ as *const _,
        data_desc.inner(),
        data_mem.ptr(),
        &beta as *const _ as *const _,
        data_desc.inner(),
        data_mem.ptr(),
    ).unwrap();
}

/// Represents the residual input of a fused operation.
/// This allows reusing the (mutable) output as input, which would normally not be allowed by Rust.    
#[derive(Debug)]
pub enum ResInput<'a> {
    Zero,
    Output,
    Other(&'a DeviceMem),
}

/// Runs `output = act(conv(input, filter) + res + bias)`.
///
/// * `res` can be 0, equal to the output or a separate tensor.
/// * `input` must be different from both `output` and `res`.
/// * `res` is assumed to have the same descriptor as `output`.
pub unsafe fn run_conv_bias_res_activation(
    handle: &mut CudnnHandle,
    activation_desc: &ActivationDescriptor,
    conv_desc: &ConvolutionDescriptor,
    algo: cudnnConvolutionFwdAlgo_t,
    work_mem: &mut DeviceMem,
    filter_desc: &FilterDescriptor,
    filter_mem: &DeviceMem,
    input_desc: &TensorDescriptor,
    input_mem: &DeviceMem,
    res: ResInput,
    bias_desc: &TensorDescriptor,
    bias_mem: &DeviceMem,
    output_desc: &TensorDescriptor,
    output_mem: &mut DeviceMem,
) {
    let alpha1: f32 = 1.0;

    assert_ne!(input_mem.ptr(), output_mem.ptr(), "input and output must be distinct");
    assert_ne!(input_mem.ptr(), bias_mem.ptr(), "input and bias must be distinct");

    // map res to actual arguments
    let (alpha2, res_mem) = match res {
        ResInput::Zero => (0f32, output_mem.ptr()),
        ResInput::Output => (1f32, output_mem.ptr()),
        ResInput::Other(mem) => (1f32, mem.ptr())
    };

    cudnnConvolutionBiasActivationForward(
        handle.inner(),
        &alpha1 as *const f32 as *const _,
        input_desc.inner(),
        input_mem.ptr(),
        filter_desc.inner(),
        filter_mem.ptr(),
        conv_desc.inner(),
        algo,
        work_mem.ptr(),
        work_mem.len(),
        &alpha2 as *const f32 as *const _,
        output_desc.inner(),
        res_mem,
        bias_desc.inner(),
        bias_mem.ptr(),
        activation_desc.inner(),
        output_desc.inner(),
        output_mem.ptr(),
    ).unwrap();
}

/// Runs `output = pool(input)`.
pub unsafe fn run_pooling(
    handle: &mut CudnnHandle,
    pool_desc: &PoolingDescriptor,
    input_desc: &TensorDescriptor,
    input_mem: &DeviceMem,
    output_desc: &TensorDescriptor,
    output_mem: &mut DeviceMem,
) {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    cudnnPoolingForward(
        handle.inner(),
        pool_desc.inner(),
        &alpha as *const _ as *const _,
        input_desc.inner(),
        input_mem.ptr(),
        &beta as *const _ as *const _,
        output_desc.inner(),
        output_mem.ptr(),
    ).unwrap();
}

/// Runs `output = op(alpha_1 * input_1, alpha_2 * input_2) + b * output`
pub unsafe fn run_tensor_op(
    handle: &mut CudnnHandle,
    op_desc: &TensorOpDescriptor,
    alpha_1: f32,
    input_1_desc: &TensorDescriptor,
    input_1_mem: &DeviceMem,
    alpha_2: f32,
    input_2_desc: &TensorDescriptor,
    input_2_mem: &DeviceMem,
    beta: f32,
    output_desc: &TensorDescriptor,
    output_mem: &DeviceMem,
) {
    cudnnOpTensor(
        handle.inner(),
        op_desc.inner(),
        &alpha_1 as *const _ as *const _,
        input_1_desc.inner(),
        input_1_mem.ptr(),
        &alpha_2 as *const _ as *const _,
        input_2_desc.inner(),
        input_2_mem.ptr(),
        &beta as *const _ as *const _,
        output_desc.inner(),
        output_mem.ptr(),
    ).unwrap();
}