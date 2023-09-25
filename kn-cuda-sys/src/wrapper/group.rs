use crate::bindings::{cublasOperation_t, cublasSgemmStridedBatched, cublasSgemm_v2, cudnnConvolutionFwdAlgo_t};
use crate::wrapper::descriptor::{
    ActivationDescriptor, ConvolutionDescriptor, FilterDescriptor, TensorDescriptor, TensorOpDescriptor,
};
use crate::wrapper::handle::{CublasHandle, CudnnHandle};
use crate::wrapper::mem::device::DevicePtr;
use crate::wrapper::operation::{run_conv_bias_res_activation, run_tensor_op};
use crate::wrapper::status::Status;

/// The arguments necessary for a fused convolution call.
#[derive(Debug)]
pub struct FusedConvolutionArgs<P = DevicePtr> {
    pub conv_desc: ConvolutionDescriptor,

    pub algo: cudnnConvolutionFwdAlgo_t,
    pub work_ptr: P,
    pub work_size_bytes: usize,

    pub filter_desc: FilterDescriptor,
    pub filter_ptr: P,
    pub input_desc: TensorDescriptor,
    pub input_ptr: P,

    pub res_ptr: Option<P>,
    pub bias_desc: TensorDescriptor,
    pub bias_ptr: P,

    pub act_desc: ActivationDescriptor,

    pub output_desc: TensorDescriptor,
    pub output_ptr: P,
}

impl FusedConvolutionArgs {
    pub unsafe fn run(&self, handle: &CudnnHandle) {
        run_conv_bias_res_activation(
            handle,
            &self.act_desc,
            &self.conv_desc,
            self.algo,
            &self.work_ptr,
            self.work_size_bytes,
            &self.filter_desc,
            &self.filter_ptr,
            &self.input_desc,
            &self.input_ptr,
            self.res_ptr.as_ref(),
            &self.bias_desc,
            &self.bias_ptr,
            &self.output_desc,
            &self.output_ptr,
        )
    }
}

#[derive(Debug)]
pub struct TensorOpArgs<P = DevicePtr> {
    pub op_desc: TensorOpDescriptor,
    pub alpha_1: f32,
    pub input_1_desc: TensorDescriptor,
    pub input_1_ptr: P,
    pub alpha_2: f32,
    pub input_2_desc: TensorDescriptor,
    pub input_2_ptr: P,
    pub beta: f32,
    pub output_desc: TensorDescriptor,
    pub output_ptr: P,
}

impl TensorOpArgs {
    pub unsafe fn run(&self, handle: &CudnnHandle) {
        run_tensor_op(
            handle,
            &self.op_desc,
            self.alpha_1,
            &self.input_1_desc,
            &self.input_1_ptr,
            self.alpha_2,
            &self.input_2_desc,
            &self.input_2_ptr,
            self.beta,
            &self.output_desc,
            &self.output_ptr,
        )
    }
}

#[derive(Debug, Clone)]
pub struct MatMulOperand<P = DevicePtr> {
    pub ptr: P,
    pub trans: cublasOperation_t,
    pub ld: i32,
    pub stride: i64,
}

impl<P> MatMulOperand<P> {
    pub fn map_ptr<K>(self, mut f: impl FnMut(P) -> K) -> MatMulOperand<K> {
        MatMulOperand {
            ptr: f(self.ptr),
            trans: self.trans,
            ld: self.ld,
            stride: self.stride,
        }
    }
}

#[derive(Debug)]
pub struct BatchedMatMulArgs<P = DevicePtr> {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub alpha: f32,
    pub beta: f32,
    pub a: MatMulOperand<P>,
    pub b: MatMulOperand<P>,
    pub c: MatMulOperand<P>,
    pub batch_count: i32,
}

impl BatchedMatMulArgs {
    /// Call `cublasSgemmStridedBatched` with the right arguments. If `batch_count == 1`, call `cublasSgemm_v2` instead.
    pub unsafe fn run(&self, handle: &CublasHandle) {
        assert_eq!(
            self.c.trans,
            cublasOperation_t::CUBLAS_OP_N,
            "A transpose output is not supported, instead you can flip and transpose the inputs"
        );

        // TODO does this actually help performance in any way?
        if self.batch_count == 1 {
            cublasSgemm_v2(
                handle.inner(),
                self.a.trans,
                self.b.trans,
                self.m,
                self.n,
                self.k,
                &(self.alpha) as *const f32,
                self.a.ptr.ptr() as *const f32,
                self.a.ld,
                self.b.ptr.ptr() as *const f32,
                self.b.ld,
                &(self.beta) as *const f32,
                self.c.ptr.ptr() as *mut f32,
                self.c.ld,
            )
            .unwrap()
        } else {
            cublasSgemmStridedBatched(
                handle.inner(),
                self.a.trans,
                self.b.trans,
                self.m,
                self.n,
                self.k,
                &(self.alpha) as *const f32,
                self.a.ptr.ptr() as *const f32,
                self.a.ld,
                self.a.stride,
                self.b.ptr.ptr() as *const f32,
                self.b.ld,
                self.b.stride,
                &(self.beta) as *const f32,
                self.c.ptr.ptr() as *mut f32,
                self.c.ld,
                self.c.stride,
                self.batch_count,
            )
            .unwrap()
        }
    }
}
