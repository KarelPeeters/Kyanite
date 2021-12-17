pub use crate::bindings::{cudnnActivationForward, cudnnAddTensor, cudnnConvolutionBiasActivationForward, cudnnConvolutionForward, cudnnConvolutionFwdAlgo_t, cudnnConvolutionFwdAlgoPerf_t, cudnnFindConvolutionForwardAlgorithm, cudnnGetConvolutionForwardAlgorithmMaxCount, cudnnStatus_t};
use crate::bindings::{cudnnActivationMode_t, cudnnOpTensor, cudnnPoolingForward};
use crate::wrapper::descriptor::{ActivationDescriptor, ConvolutionDescriptor, FilterDescriptor, PoolingDescriptor, TensorDescriptor, TensorOpDescriptor};
use crate::wrapper::handle::CudnnHandle;
use crate::wrapper::mem::device::DeviceMem;
use crate::wrapper::status::Status;

//TODO try automatic conv benchmarking thing again
//  careful, cudnnConvolutionBiasActivationForward may require this algorithm
pub const STANDARD_CONV_ALGO: cudnnConvolutionFwdAlgo_t =
    cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

pub fn find_conv_algorithms(
    handle: &CudnnHandle,
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
    handle: &CudnnHandle,
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
    assert_eq!(input_desc.size_bytes(), input_mem.len_bytes());
    assert_eq!(filter_desc.size_bytes(), filter_mem.len_bytes());
    assert_eq!(output_desc.size_bytes(), output_mem.len_bytes());

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
        work_mem.len_bytes(),
        &beta as *const _ as *const _,
        output_desc.inner(),
        output_mem.ptr(),
    ).unwrap();
}

/// Run `output += input`. `input` can have dimensions of size 1 which are broadcasted to the shape of `output`.
pub unsafe fn run_add_tensor(
    handle: &CudnnHandle,
    input_desc: &TensorDescriptor,
    input_mem: &DeviceMem,
    output_desc: &TensorDescriptor,
    output_mem: &DeviceMem,
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
    handle: &CudnnHandle,
    activation_desc: &ActivationDescriptor,
    input_desc: &TensorDescriptor,
    input_mem: &DeviceMem,
    output_desc: &TensorDescriptor,
    output_mem: &DeviceMem,
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
    handle: &CudnnHandle,
    activation_desc: &ActivationDescriptor,
    data_desc: &TensorDescriptor,
    data_mem: &DeviceMem,
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

/// Runs `output = act(conv(input, filter) + res + bias)`.
///
/// * `res` can be 0, equal to the output or a separate tensor.
/// * `input` must be different from both `output` and `res`.
/// * `res` is assumed to have the same descriptor as `output`.
pub unsafe fn run_conv_bias_res_activation(
    handle: &CudnnHandle,
    activation_desc: &ActivationDescriptor,
    conv_desc: &ConvolutionDescriptor,
    algo: cudnnConvolutionFwdAlgo_t,
    work_mem: &DeviceMem,
    filter_desc: &FilterDescriptor,
    filter_mem: &DeviceMem,
    input_desc: &TensorDescriptor,
    input_mem: &DeviceMem,
    res_mem: Option<&DeviceMem>,
    bias_desc: &TensorDescriptor,
    bias_mem: &DeviceMem,
    output_desc: &TensorDescriptor,
    output_mem: &DeviceMem,
) {
    let alpha1: f32 = 1.0;

    // map res to actual arguments
    let (alpha2, res_ptr) = match res_mem {
        None => (0f32, output_mem.ptr()),
        Some(res_mem) => (1f32, res_mem.ptr()),
    };

    assert_ne!(input_mem.ptr(), output_mem.ptr(), "input and output must be distinct");
    assert_ne!(input_mem.ptr(), bias_mem.ptr(), "input and bias must be distinct");
    assert_ne!(input_mem.ptr(), res_ptr, "input and res must be distinct");
    assert_eq!(bias_desc.shape()[0], 1, "bias first dim must be 1");
    assert_eq!(bias_desc.shape()[1], output_desc.shape()[1], "bias channels must match output channels");
    assert_eq!(bias_desc.strides()[1], 1, "bias second stride must be one");
    assert_eq!(&conv_desc.output_shape(input_desc, filter_desc), output_desc.shape(), "output shape mismatch");

    assert_eq!(input_desc.rank(), 4, "input desc wrong rank");
    assert_eq!(output_desc.rank(), 4, "output desc wrong rank");
    assert_eq!(bias_desc.rank(), 4, "bias desc wrong rank");

    assert!(
        activation_desc.mode() == cudnnActivationMode_t::CUDNN_ACTIVATION_RELU ||
            activation_desc.mode() == cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY,
        "unsupported activation mode"
    );

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
        work_mem.len_bytes(),
        &alpha2 as *const f32 as *const _,
        output_desc.inner(),
        res_ptr,
        bias_desc.inner(),
        bias_mem.ptr(),
        activation_desc.inner(),
        output_desc.inner(),
        output_mem.ptr(),
    ).unwrap();
}

/// Runs `output = pool(input)`.
pub unsafe fn run_pooling(
    handle: &CudnnHandle,
    pool_desc: &PoolingDescriptor,
    input_desc: &TensorDescriptor,
    input_mem: &DeviceMem,
    output_desc: &TensorDescriptor,
    output_mem: &DeviceMem,
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
    handle: &CudnnHandle,
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