pub use crate::bindings::{cudnnActivationForward, cudnnAddTensor, cudnnConvolutionBiasActivationForward, cudnnConvolutionForward, cudnnConvolutionFwdAlgo_t, cudnnConvolutionFwdAlgoPerfStruct, cudnnFindConvolutionForwardAlgorithm, cudnnGetConvolutionForwardAlgorithmMaxCount, cudnnStatus_t};
use crate::bindings::cudnnPoolingForward;
use crate::wrapper::descriptor::{ActivationDescriptor, ConvolutionDescriptor, FilterDescriptor, PoolingDescriptor, TensorDescriptor};
use crate::wrapper::handle::CudnnHandle;
use crate::wrapper::mem::DeviceMem;
use crate::wrapper::status::Status;

pub fn find_conv_algorithms(
    handle: &mut CudnnHandle,
    conv: &ConvolutionDescriptor,
    filter: &FilterDescriptor,
    input: &TensorDescriptor,
    output: &TensorDescriptor,
) -> Vec<cudnnConvolutionFwdAlgoPerfStruct> {
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

// TODO the &mut stuff for all of these functions is kind of sketchy, since they don't really modify the mem yet.
//   They just schedule the memory to be modified by the stream inside the handle.

/// Run `output = conv(input, filter)`
pub fn run_conv(
    handle: &mut CudnnHandle,
    conv_desc: &ConvolutionDescriptor,
    algo: cudnnConvolutionFwdAlgo_t,
    work_mem: &mut DeviceMem,
    filter_desc: &FilterDescriptor,
    filter_mem: &DeviceMem,
    input_desc: &TensorDescriptor,
    input_mem: &DeviceMem,
    output_desc: &TensorDescriptor,
    output_mem: &mut DeviceMem,
) {
    assert_eq!(input_desc.size(), input_mem.size());
    assert_eq!(filter_desc.size(), filter_mem.size());
    assert_eq!(output_desc.size(), output_mem.size());

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
            algo,
            work_mem.inner(),
            work_mem.size(),
            &beta as *const _ as *const _,
            output_desc.inner(),
            output_mem.inner(),
        ).unwrap();
    }
}

/// Run `output += input`. `output` can have dimensions of size 1 which are broadcasted to the shape of `output`.
pub fn run_add_tensor(
    handle: &mut CudnnHandle,
    input_desc: &TensorDescriptor,
    input_mem: &DeviceMem,
    output_desc: &TensorDescriptor,
    output_mem: &mut DeviceMem,
) {
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;

    unsafe {
        cudnnAddTensor(
            handle.inner(),
            &alpha as *const _ as *const _,
            input_desc.inner(),
            input_mem.inner(),
            &beta as *const _ as *const _,
            output_desc.inner(),
            output_mem.inner(),
        ).unwrap();
    }
}

/// Run `output = act(input)`.
pub fn run_activation(
    handle: &mut CudnnHandle,
    activation_desc: &ActivationDescriptor,
    input_desc: &TensorDescriptor,
    input_mem: &DeviceMem,
    output_desc: &TensorDescriptor,
    output_mem: &mut DeviceMem,
) {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    unsafe {
        cudnnActivationForward(
            handle.inner(),
            activation_desc.inner(),
            &alpha as *const _ as *const _,
            input_desc.inner(),
            input_mem.inner(),
            &beta as *const _ as *const _,
            output_desc.inner(),
            output_mem.inner(),
        ).unwrap();
    }
}

/// Runs `output = act(output)`.
pub fn run_activation_in_place(
    handle: &mut CudnnHandle,
    activation_desc: &ActivationDescriptor,
    data_desc: &TensorDescriptor,
    data_mem: &mut DeviceMem,
) {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    unsafe {
        cudnnActivationForward(
            handle.inner(),
            activation_desc.inner(),
            &alpha as *const _ as *const _,
            data_desc.inner(),
            data_mem.inner(),
            &beta as *const _ as *const _,
            data_desc.inner(),
            data_mem.inner(),
        ).unwrap();
    }
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
pub fn run_conv_bias_res_activation(
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

    unsafe {
        assert_ne!(input_mem.inner(), output_mem.inner(), "input and output must be distinct");
        assert_ne!(input_mem.inner(), bias_mem.inner(), "input and bias must be distinct");

        // map res to actual arguments
        let (alpha2,  res_mem) = match res {
            ResInput::Zero => (0f32, output_mem.inner()),
            ResInput::Output => (1f32, output_mem.inner()),
            ResInput::Other(mem) => (1f32, mem.inner())
        };

        cudnnConvolutionBiasActivationForward(
            handle.inner(),
            &alpha1 as *const f32 as *const _,
            input_desc.inner(),
            input_mem.inner(),
            filter_desc.inner(),
            filter_mem.inner(),
            conv_desc.inner(),
            algo,
            work_mem.inner(),
            work_mem.size(),
            &alpha2 as *const f32 as *const _,
            output_desc.inner(),
            res_mem,
            bias_desc.inner(),
            bias_mem.inner(),
            activation_desc.inner(),
            output_desc.inner(),
            output_mem.inner(),
        ).unwrap();
    }
}

/// Runs `output = pool(input)`.
pub fn run_pooling(
    handle: &mut CudnnHandle,
    pool_desc: &PoolingDescriptor,
    input_desc: &TensorDescriptor,
    input_mem: &DeviceMem,
    output_desc: &TensorDescriptor,
    output_mem: &mut DeviceMem,
) {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    unsafe {
        cudnnPoolingForward(
            handle.inner(),
            pool_desc.inner(),
            &alpha as *const _ as *const _,
            input_desc.inner(),
            input_mem.inner(),
            &beta as *const _ as *const _,
            output_desc.inner(),
            output_mem.inner(),
        ).unwrap();
    }
}