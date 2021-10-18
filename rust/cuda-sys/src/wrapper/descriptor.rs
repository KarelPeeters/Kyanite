use std::ptr::null_mut;

use crate::bindings::{cudnnActivationDescriptor_t, cudnnActivationMode_t, cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, cudnnConvolutionMode_t, cudnnCreateActivationDescriptor, cudnnCreateConvolutionDescriptor, cudnnCreateFilterDescriptor, cudnnCreateOpTensorDescriptor, cudnnCreatePoolingDescriptor, cudnnCreateTensorDescriptor, cudnnDataType_t, cudnnDestroyActivationDescriptor, cudnnDestroyConvolutionDescriptor, cudnnDestroyFilterDescriptor, cudnnDestroyOpTensorDescriptor, cudnnDestroyPoolingDescriptor, cudnnDestroyTensorDescriptor, cudnnFilterDescriptor_t, cudnnGetConvolution2dForwardOutputDim, cudnnGetConvolutionForwardWorkspaceSize, cudnnGetFilterSizeInBytes, cudnnGetPooling2dForwardOutputDim, cudnnGetTensorSizeInBytes, cudnnNanPropagation_t, cudnnOpTensorDescriptor_t, cudnnOpTensorOp_t, cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnSetActivationDescriptor, cudnnSetConvolution2dDescriptor, cudnnSetFilter4dDescriptor, cudnnSetOpTensorDescriptor, cudnnSetPooling2dDescriptor, cudnnSetTensorNdDescriptor, cudnnTensorDescriptor_t, cudnnTensorFormat_t};
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
        unsafe { cudnnDestroyTensorDescriptor(self.inner).unwrap() }
    }
}

impl TensorDescriptor {
    pub fn new(shape: Vec<i32>, strides: Vec<i32>) -> Self {
        //TODO maybe re-enable this assert if we actually run into issues
        assert!(shape.len() > 2, "Tensors must be at least 3d, got shape {:?} strides {:?}", shape, strides);

        let rank = shape.len();
        assert_eq!(rank, strides.len());

        for i in 0..rank {
            assert_ne!(shape[i], 0, "Zero-sized dimensions are not allowed");
            assert!(strides[i] > 0);
        }

        unsafe {
            let mut inner = null_mut();
            cudnnCreateTensorDescriptor(&mut inner as *mut _).unwrap();
            cudnnSetTensorNdDescriptor(
                inner,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
                rank as i32,
                shape.as_ptr(),
                strides.as_ptr(),
            ).unwrap();

            TensorDescriptor { inner, shape, strides }
        }
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
        unsafe {
            cudnnDestroyFilterDescriptor(self.inner).unwrap()
        }
    }
}

impl FilterDescriptor {
    /// * `k`: output channels
    /// * `c`: input channels
    /// * `(h, w)`: kernel size
    pub fn new(k: i32, c: i32, h: i32, w: i32) -> Self {
        //TODO whats with (h, w) here? that's super inconsistent with everything else?
        assert_eq!(h, w, "Only square kernels supported for now");

        unsafe {
            let mut inner = null_mut();
            cudnnCreateFilterDescriptor(&mut inner as *mut _).unwrap();
            cudnnSetFilter4dDescriptor(
                inner,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
                cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                k, c, h, w,
            ).unwrap();
            FilterDescriptor { inner, shape: [k, c, h, w] }
        }
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
        unsafe { cudnnDestroyConvolutionDescriptor(self.0).unwrap() }
    }
}

impl ConvolutionDescriptor {
    pub fn new(
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_v: i32,
        dilation_h: i32,
        dilation_w: i32,
    ) -> Self {
        let checked = [stride_h, stride_v, dilation_h, dilation_w];
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
                pad_h, pad_w, stride_h, stride_v, dilation_h, dilation_w,
                cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
                cudnnDataType_t::CUDNN_DATA_FLOAT,
            ).unwrap();
            ConvolutionDescriptor(inner)
        }
    }

    pub fn workspace_size(
        &self,
        handle: &mut CudnnHandle,
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
            ).unwrap();
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
            ).unwrap();
            [n, c, h, w]
        }
    }

    pub unsafe fn inner(&self) -> cudnnConvolutionDescriptor_t {
        self.0
    }
}

#[derive(Debug)]
pub struct ActivationDescriptor(cudnnActivationDescriptor_t);

impl Drop for ActivationDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyActivationDescriptor(self.0).unwrap() }
    }
}

impl ActivationDescriptor {
    pub fn new(
        mode: cudnnActivationMode_t,
        coef: f32,
    ) -> Self {
        unsafe {
            let mut inner = null_mut();
            cudnnCreateActivationDescriptor(&mut inner as *mut _).unwrap();
            cudnnSetActivationDescriptor(
                inner,
                mode,
                cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
                coef as f64,
            ).unwrap();
            ActivationDescriptor(inner)
        }
    }

    pub unsafe fn inner(&self) -> cudnnActivationDescriptor_t {
        self.0
    }
}

pub struct PoolingDescriptor(cudnnPoolingDescriptor_t);

impl Drop for PoolingDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyPoolingDescriptor(self.0).unwrap() }
    }
}

impl PoolingDescriptor {
    pub fn new(
        mode: cudnnPoolingMode_t,
        h: i32,
        w: i32,
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_v: i32,
    ) -> Self {
        unsafe {
            let mut inner = null_mut();
            cudnnCreatePoolingDescriptor(&mut inner as *mut _).unwrap();
            cudnnSetPooling2dDescriptor(
                inner,
                mode,
                cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
                h, w,
                pad_h, pad_w,
                stride_h, stride_v,
            ).unwrap();
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
            ).unwrap();
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
    operation: cudnnOpTensorOp_t,
}

impl Drop for TensorOpDescriptor {
    fn drop(&mut self) {
        unsafe {
            cudnnDestroyOpTensorDescriptor(self.inner).unwrap()
        }
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
            ).unwrap();

            TensorOpDescriptor { inner, operation }
        }
    }

    pub unsafe fn inner(&self) -> cudnnOpTensorDescriptor_t {
        self.inner
    }
}
