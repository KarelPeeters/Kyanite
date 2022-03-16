use std::ptr::null_mut;

use crate::bindings::{
    cudnnActivationDescriptor_t, cudnnActivationMode_t, cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t,
    cudnnConvolutionMode_t, cudnnCreateActivationDescriptor, cudnnCreateConvolutionDescriptor,
    cudnnCreateFilterDescriptor, cudnnCreateOpTensorDescriptor, cudnnCreatePoolingDescriptor,
    cudnnCreateTensorDescriptor, cudnnDataType_t, cudnnDestroyActivationDescriptor, cudnnDestroyConvolutionDescriptor,
    cudnnDestroyFilterDescriptor, cudnnDestroyOpTensorDescriptor, cudnnDestroyPoolingDescriptor,
    cudnnDestroyTensorDescriptor, cudnnFilterDescriptor_t, cudnnGetConvolution2dForwardOutputDim,
    cudnnGetConvolutionForwardWorkspaceSize, cudnnGetFilterSizeInBytes, cudnnGetPooling2dForwardOutputDim,
    cudnnGetTensorSizeInBytes, cudnnNanPropagation_t, cudnnOpTensorDescriptor_t, cudnnOpTensorOp_t,
    cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnSetActivationDescriptor, cudnnSetConvolution2dDescriptor,
    cudnnSetFilter4dDescriptor, cudnnSetOpTensorDescriptor, cudnnSetPooling2dDescriptor, cudnnSetTensorNdDescriptor,
    cudnnTensorDescriptor_t, cudnnTensorFormat_t,
};
use crate::wrapper::handle::CudnnHandle;
use crate::wrapper::status::Status;

#[derive(Debug)]
pub struct TensorDescriptor {
    inner: cudnnTensorDescriptor_t,
    shape: Vec<i32>,
    strides: Vec<i32>,
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyTensorDescriptor(self.inner).unwrap_in_drop() }
    }
}

impl TensorDescriptor {
    pub fn new(shape: Vec<i32>, strides: Vec<i32>) -> Self {
        Self::new_with_type(shape, strides, cudnnDataType_t::CUDNN_DATA_FLOAT)
    }

    pub fn new_with_type(shape: Vec<i32>, strides: Vec<i32>, data_type: cudnnDataType_t) -> Self {
        assert!(
            shape.len() >= 4,
            "Tensors must be at least 4d, got shape {:?} strides {:?}",
            shape,
            strides
        );

        let rank = shape.len();
        assert_eq!(rank, strides.len());

        assert!(
            shape.iter().all(|&x| x > 0),
            "Shape cannot be negative, got shape {:?} with strides {:?}",
            shape,
            strides,
        );

        unsafe {
            let mut inner = null_mut();
            cudnnCreateTensorDescriptor(&mut inner as *mut _).unwrap();
            cudnnSetTensorNdDescriptor(inner, data_type, rank as i32, shape.as_ptr(), strides.as_ptr()).unwrap();

            TensorDescriptor { inner, shape, strides }
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &[i32] {
        &self.shape
    }

    pub fn strides(&self) -> &[i32] {
        &self.strides
    }

    pub unsafe fn inner(&self) -> cudnnTensorDescriptor_t {
        self.inner
    }

    pub fn size_bytes(&self) -> usize {
        unsafe {
            let mut result = 0;
            cudnnGetTensorSizeInBytes(self.inner, &mut result as *mut _).unwrap();
            result
        }
    }
}

#[derive(Debug)]
pub struct FilterDescriptor {
    inner: cudnnFilterDescriptor_t,
    shape: [i32; 4],
}

impl Drop for FilterDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyFilterDescriptor(self.inner).unwrap_in_drop() }
    }
}

impl FilterDescriptor {
    pub fn new_with_type_format(
        k: i32,
        c: i32,
        h: i32,
        w: i32,
        data_type: cudnnDataType_t,
        format: cudnnTensorFormat_t,
    ) -> Self {
        unsafe {
            let mut inner = null_mut();
            cudnnCreateFilterDescriptor(&mut inner as *mut _).unwrap();
            cudnnSetFilter4dDescriptor(inner, data_type, format, k, c, h, w).unwrap();
            FilterDescriptor {
                inner,
                shape: [k, c, h, w],
            }
        }
    }

    /// * `k`: output channels
    /// * `c`: input channels
    /// * `(h, w)`: kernel size
    pub fn new(k: i32, c: i32, h: i32, w: i32) -> Self {
        Self::new_with_type_format(
            k,
            c,
            h,
            w,
            cudnnDataType_t::CUDNN_DATA_FLOAT,
            cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        )
    }

    pub unsafe fn inner(&self) -> cudnnFilterDescriptor_t {
        self.inner
    }

    pub fn shape(&self) -> [i32; 4] {
        self.shape
    }

    pub fn size_bytes(&self) -> usize {
        unsafe {
            let mut result = 0;
            cudnnGetFilterSizeInBytes(self.inner, &mut result as *mut _).unwrap();
            result
        }
    }
}

#[derive(Debug)]
pub struct ConvolutionDescriptor(cudnnConvolutionDescriptor_t);

impl Drop for ConvolutionDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyConvolutionDescriptor(self.0).unwrap_in_drop() }
    }
}

impl ConvolutionDescriptor {
    pub fn new(pad_y: i32, pad_x: i32, stride_y: i32, stride_x: i32, dilation_y: i32, dilation_x: i32) -> Self {
        assert!(
            pad_y >= 0 && pad_x >= 0,
            "Padding cannot be negative, got ({}, {})",
            pad_y,
            pad_x
        );

        let checked = [stride_y, stride_x, dilation_y, dilation_x];
        assert!(
            checked.iter().all(|&x| x > 0),
            "Dilation and stride must be strictly positive, 1 means dense convolution. Got {:?}",
            checked
        );

        unsafe {
            let mut inner = null_mut();
            cudnnCreateConvolutionDescriptor(&mut inner as *mut _).unwrap();
            cudnnSetConvolution2dDescriptor(
                inner,
                pad_y,
                pad_x,
                stride_y,
                stride_x,
                dilation_y,
                dilation_x,
                cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
            )
            .unwrap();
            ConvolutionDescriptor(inner)
        }
    }

    pub fn workspace_size(
        &self,
        handle: &CudnnHandle,
        algo: cudnnConvolutionFwdAlgo_t,
        input: &TensorDescriptor,
        filter: &FilterDescriptor,
        output: &TensorDescriptor,
    ) -> usize {
        assert_eq!(
            &self.output_shape(input, filter)[..],
            &output.shape,
            "Output shape mismatch"
        );

        let mut workspace: usize = 0;

        unsafe {
            cudnnGetConvolutionForwardWorkspaceSize(
                handle.inner(),
                input.inner,
                filter.inner,
                self.inner(),
                output.inner(),
                algo,
                &mut workspace as *mut _,
            )
            .unwrap();
        }

        workspace
    }

    pub fn output_shape(&self, input_desc: &TensorDescriptor, filter_desc: &FilterDescriptor) -> [i32; 4] {
        assert_eq!(
            input_desc.shape[1], filter_desc.shape[1],
            "Input channel count mismatch, input {:?} filter {:?}",
            input_desc.shape, filter_desc.shape
        );

        unsafe {
            let mut n = 0;
            let mut c = 0;
            let mut h = 0;
            let mut w = 0;
            cudnnGetConvolution2dForwardOutputDim(
                self.inner(),
                input_desc.inner(),
                filter_desc.inner(),
                &mut n as *mut _,
                &mut c as *mut _,
                &mut h as *mut _,
                &mut w as *mut _,
            )
            .unwrap();
            [n, c, h, w]
        }
    }

    pub unsafe fn inner(&self) -> cudnnConvolutionDescriptor_t {
        self.0
    }
}

#[derive(Debug)]
pub struct ActivationDescriptor {
    inner: cudnnActivationDescriptor_t,
    mode: cudnnActivationMode_t,
}

impl Drop for ActivationDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyActivationDescriptor(self.inner).unwrap_in_drop() }
    }
}

impl ActivationDescriptor {
    pub fn new(mode: cudnnActivationMode_t, coef: f32) -> Self {
        unsafe {
            let mut inner = null_mut();
            cudnnCreateActivationDescriptor(&mut inner as *mut _).unwrap();
            cudnnSetActivationDescriptor(inner, mode, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN, coef as f64)
                .unwrap();
            ActivationDescriptor { inner, mode }
        }
    }

    pub unsafe fn inner(&self) -> cudnnActivationDescriptor_t {
        self.inner
    }

    pub fn mode(&self) -> cudnnActivationMode_t {
        self.mode
    }
}

#[derive(Debug)]
pub struct PoolingDescriptor(cudnnPoolingDescriptor_t);

impl Drop for PoolingDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyPoolingDescriptor(self.0).unwrap_in_drop() }
    }
}

impl PoolingDescriptor {
    pub fn new(
        mode: cudnnPoolingMode_t,
        size_y: i32,
        size_x: i32,
        pad_y: i32,
        pad_x: i32,
        stride_y: i32,
        stride_x: i32,
    ) -> Self {
        unsafe {
            let mut inner = null_mut();
            cudnnCreatePoolingDescriptor(&mut inner as *mut _).unwrap();
            cudnnSetPooling2dDescriptor(
                inner,
                mode,
                cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
                size_y,
                size_x,
                pad_y,
                pad_x,
                stride_y,
                stride_x,
            )
            .unwrap();
            PoolingDescriptor(inner)
        }
    }

    pub fn output_shape(&self, input_desc: &TensorDescriptor) -> [i32; 4] {
        unsafe {
            let mut n = 0;
            let mut c = 0;
            let mut h = 0;
            let mut w = 0;
            cudnnGetPooling2dForwardOutputDim(
                self.inner(),
                input_desc.inner(),
                &mut n as *mut _,
                &mut c as *mut _,
                &mut h as *mut _,
                &mut w as *mut _,
            )
            .unwrap();
            [n, c, h, w]
        }
    }

    pub unsafe fn inner(&self) -> cudnnPoolingDescriptor_t {
        self.0
    }
}

#[derive(Debug)]
pub struct TensorOpDescriptor {
    inner: cudnnOpTensorDescriptor_t,
    #[allow(dead_code)]
    operation: cudnnOpTensorOp_t,
}

impl Drop for TensorOpDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyOpTensorDescriptor(self.inner).unwrap_in_drop() }
    }
}

impl TensorOpDescriptor {
    pub fn new(operation: cudnnOpTensorOp_t) -> Self {
        unsafe {
            let mut inner = null_mut();
            cudnnCreateOpTensorDescriptor(&mut inner as *mut _).unwrap();
            cudnnSetOpTensorDescriptor(
                inner,
                operation,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
                cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
            )
            .unwrap();

            TensorOpDescriptor { inner, operation }
        }
    }

    pub unsafe fn inner(&self) -> cudnnOpTensorDescriptor_t {
        self.inner
    }
}
