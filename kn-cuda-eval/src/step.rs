use std::ops::ControlFlow;

use internal_iterator::InternalIterator;
use itertools::zip_eq;

use kn_cuda_sys::wrapper::group::{BatchedMatMulArgs, FusedConvolutionArgs};
use kn_cuda_sys::wrapper::handle::{CublasHandle, CudaStream, CudnnHandle, Device};
use kn_cuda_sys::wrapper::mem::device::DevicePtr;
use kn_cuda_sys::wrapper::rtc::core::CuFunction;
use kn_graph::graph::Value;

use crate::autokernel::gather::GatherKernel;
use crate::autokernel::layernorm::LayernormKernel;
use crate::autokernel::reduce::ReduceKernel;
use crate::autokernel::scalar::ScalarKernel;
use crate::autokernel::softmax::SoftmaxKernel;
use crate::offset_tensor::PtrTensor;

#[derive(Debug)]
pub struct StepInfo<K, P> {
    pub debug_value: Value,
    pub debug_id: String,
    pub step: Step<K, P>,
}

#[derive(Debug)]
pub enum Step<K, P> {
    Conv(FusedConvolutionArgs<P>),
    MatMul(BatchedMatMulArgs<P>),
    ScalarOp(ScalarOpArgs<K, P>),
    ReduceOp(ReduceOpArgs<K, P>),
    SoftmaxOp(SoftmaxOpArgs<K, P>),
    LayernormOp(LayernormOpArgs<K, P>),
    GatherOp(GatherOpArgs<K, P>),
}

#[derive(Debug)]
pub struct ScalarOpArgs<K, P> {
    pub kernel: ScalarKernel<K>,
    pub operands: Vec<PtrTensor<P>>,
    pub operand_kinds: Vec<OperandKind>,
}

#[derive(Debug)]
pub struct ReduceOpArgs<K, P> {
    pub kernel: ReduceKernel<K>,
    pub input: PtrTensor<P>,
    pub output: PtrTensor<P>,
}

#[derive(Debug)]
pub struct SoftmaxOpArgs<K, P> {
    pub kernel: SoftmaxKernel<K>,
    pub input: PtrTensor<P>,
    pub output: PtrTensor<P>,
}

#[derive(Debug)]
pub struct GatherOpArgs<K, P> {
    pub kernel: GatherKernel<K>,
    pub input: PtrTensor<P>,
    pub indices: PtrTensor<P>,
    pub output: PtrTensor<P>,
}

#[derive(Debug)]
pub struct LayernormOpArgs<K, P> {
    pub kernel: LayernormKernel<K>,
    pub input0: PtrTensor<P>,
    pub input1: Option<PtrTensor<P>>,
    pub output: PtrTensor<P>,
}

#[derive(Debug)]
pub struct PlanStepOperands<'a, K, P>(&'a Step<K, P>);

#[derive(Debug)]
pub struct Operand<T> {
    pub kind: OperandKind,
    pub value: T,
}

// TODO think about how to properly represent all of this: read/write/clobber/read+write/...
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum OperandKind {
    /// Memory is read, nothing is written.
    In,
    /// Memory is written, nothing is read.
    Out,
    /// Memory is both read and written.
    InOut,
    /// Memory is used as scratch space, it is both written to and read from but only is that order.
    /// This means that any pre-existing data in the memory doesn't matter.
    Scratch,
}

//TODO is there some way to reduce the huge amount of boilerplate here?
impl<K, P> Step<K, P> {
    pub fn ptr_operands(&self) -> PlanStepOperands<K, P> {
        PlanStepOperands(self)
    }

    pub fn map_kernel_and_ptrs<L, T>(self, mut f_ker: impl FnMut(&K) -> L, mut f_ptr: impl FnMut(P) -> T) -> Step<L, T> {
        match self {
            Step::Conv(args) => Step::Conv(FusedConvolutionArgs {
                conv_desc: args.conv_desc,
                algo: args.algo,
                work_ptr: f_ptr(args.work_ptr),
                work_size_bytes: args.work_size_bytes,
                filter_desc: args.filter_desc,
                filter_ptr: f_ptr(args.filter_ptr),
                input_desc: args.input_desc,
                input_ptr: f_ptr(args.input_ptr),
                res_ptr: args.res_ptr.map(&mut f_ptr),
                bias_desc: args.bias_desc,
                bias_ptr: f_ptr(args.bias_ptr),
                act_desc: args.act_desc,
                output_desc: args.output_desc,
                output_ptr: f_ptr(args.output_ptr),
            }),
            Step::MatMul(args) => Step::MatMul(BatchedMatMulArgs {
                m: args.m,
                n: args.n,
                k: args.k,
                alpha: args.alpha,
                beta: args.beta,
                a: args.a.map_ptr(&mut f_ptr),
                b: args.b.map_ptr(&mut f_ptr),
                c: args.c.map_ptr(&mut f_ptr),
                batch_count: args.batch_count,
            }),
            Step::ScalarOp(args) => Step::ScalarOp(ScalarOpArgs {
                kernel: args.kernel.map_kernel(&mut f_ker),
                operands: args.operands.into_iter().map(|op| op.map_ptr(&mut f_ptr)).collect(),
                operand_kinds: args.operand_kinds.clone(),
            }),
            Step::ReduceOp(args) => Step::ReduceOp(ReduceOpArgs {
                kernel: args.kernel.map_kernel(&mut f_ker),
                input: args.input.map_ptr(&mut f_ptr),
                output: args.output.map_ptr(&mut f_ptr),
            }),
            Step::SoftmaxOp(args) => Step::SoftmaxOp(SoftmaxOpArgs {
                kernel: args.kernel.map_kernel(&mut f_ker),
                input: args.input.map_ptr(&mut f_ptr),
                output: args.output.map_ptr(&mut f_ptr),
            }),
            Step::LayernormOp(args) => Step::LayernormOp(LayernormOpArgs {
                kernel: args.kernel.map_kernel(&mut f_ker),
                input0: args.input0.map_ptr(&mut f_ptr),
                input1: args.input1.map(|op| op.map_ptr(&mut f_ptr)),
                output: args.output.map_ptr(&mut f_ptr),
            }),
            Step::GatherOp(args) => Step::GatherOp(GatherOpArgs {
                kernel: args.kernel.map_kernel(&mut f_ker),
                input: args.input.map_ptr(&mut f_ptr),
                indices: args.indices.map_ptr(&mut f_ptr),
                output: args.output.map_ptr(&mut f_ptr),
            }),
        }
    }

    // TODO report elide lifetimes issue in rust plugin
    fn for_each_ptr<'a, R>(&'a self, mut f: impl FnMut(Operand<&'a P>) -> ControlFlow<R>) -> ControlFlow<R> {
        match self {
            Step::Conv(FusedConvolutionArgs {
                conv_desc: _,
                algo: _,
                work_ptr,
                work_size_bytes: _,
                filter_desc: _,
                filter_ptr,
                input_desc: _,
                input_ptr,
                res_ptr,
                bias_desc: _,
                bias_ptr,
                act_desc: _,
                output_desc: _,
                output_ptr,
            }) => {
                f(Operand::new_scratch(work_ptr))?;
                f(Operand::new_in(filter_ptr))?;
                f(Operand::new_in(input_ptr))?;
                f(Operand::new_in(bias_ptr))?;
                if let Some(res_ptr) = res_ptr {
                    f(Operand::new_in(res_ptr))?;
                }
                f(Operand::new_out(output_ptr))?;
            }
            Step::MatMul(BatchedMatMulArgs {
                m: _,
                n: _,
                k: _,
                alpha: _,
                beta,
                a,
                b,
                c,
                batch_count: _,
            }) => {
                f(Operand::new_in(&a.ptr))?;
                f(Operand::new_in(&b.ptr))?;

                if *beta == 0.0 {
                    f(Operand::new_out(&c.ptr))?;
                } else {
                    f(Operand::new_real_inout(&c.ptr))?;
                }
            }
            Step::ScalarOp(ScalarOpArgs { kernel: _, operands, operand_kinds }) => {
                zip_eq(operands, operand_kinds).try_for_each(|(op, &kind)| f(Operand { value: op.ptr(), kind }))?
            }
            Step::ReduceOp(ReduceOpArgs {
                kernel: _,
                input,
                output,
            }) => {
                f(Operand::new_in(input.ptr()))?;
                f(Operand::new_out(output.ptr()))?;
            }
            Step::SoftmaxOp(SoftmaxOpArgs {
                kernel: _,
                input,
                output,
            }) => {
                f(Operand::new_in(input.ptr()))?;
                f(Operand::new_out(output.ptr()))?;
            }
            Step::LayernormOp(LayernormOpArgs {
                kernel: _,
                input0,
                input1,
                output,
            }) => {
                f(Operand::new_in(input0.ptr()))?;
                if let Some(input1) = input1 {
                    f(Operand::new_in(input1.ptr()))?;
                }
                f(Operand::new_out(output.ptr()))?;
            }
            Step::GatherOp(GatherOpArgs {
                kernel: _,
                input,
                indices,
                output,
            }) => {
                f(Operand::new_in(input.ptr()))?;
                f(Operand::new_in(indices.ptr()))?;
                f(Operand::new_out(output.ptr()))?;
            }
        }

        ControlFlow::Continue(())
    }
}

impl<'a, K, P> InternalIterator for PlanStepOperands<'a, K, P> {
    type Item = Operand<&'a P>;

    fn try_for_each<R, F>(self, f: F) -> ControlFlow<R>
    where
        F: FnMut(Self::Item) -> ControlFlow<R>,
    {
        self.0.for_each_ptr(f)
    }
}

impl<T> Operand<T> {
    pub fn new_in(value: T) -> Self {
        Operand {
            kind: OperandKind::In,
            value,
        }
    }

    pub fn new_out(value: T) -> Self {
        Operand {
            kind: OperandKind::Out,
            value,
        }
    }

    // TODO rename back to inout
    pub fn new_real_inout(value: T) -> Self {
        Operand {
            kind: OperandKind::InOut,
            value,
        }
    }

    pub fn new_scratch(value: T) -> Self {
        Operand {
            kind: OperandKind::Scratch,
            value,
        }
    }

    #[allow(dead_code)]
    pub fn as_ref(&self) -> Operand<&T> {
        Operand {
            kind: self.kind,
            value: &self.value,
        }
    }

    pub fn map_value<K>(self, mut f: impl FnMut(T) -> K) -> Operand<K> {
        Operand {
            kind: self.kind,
            value: f(self.value),
        }
    }
}

#[derive(Debug)]
pub struct Handles {
    pub cudnn: CudnnHandle,
    pub cublas: CublasHandle,
}

impl Handles {
    pub fn device(&self) -> Device {
        self.stream().device()
    }

    pub fn stream(&self) -> &CudaStream {
        // prefer the cudnn stream, make sure to synchronize when using cublas
        self.cudnn.stream()
    }

    pub fn sync_before_cublas(&self) {
        let event_before = self.cudnn.stream().record_event();
        self.cublas.stream().wait_for_event(&event_before);
    }

    pub fn sync_after_cublas(&self) {
        let event_after = self.cublas.stream().record_event();
        self.cudnn.stream().wait_for_event(&event_after);
    }
}

impl Step<CuFunction, DevicePtr> {
    pub unsafe fn run(&self, handles: &Handles) {
        match self {
            Step::Conv(args) => args.run(&handles.cudnn),
            Step::MatMul(args) => {
                handles.sync_before_cublas();
                args.run(&handles.cublas);
                handles.sync_after_cublas();
            }
            Step::ScalarOp(args) => args.run(handles.stream()),
            Step::ReduceOp(args) => args.run(handles.stream()),
            Step::SoftmaxOp(args) => args.run(handles.stream()),
            Step::LayernormOp(args) => args.run(handles.stream()),
            Step::GatherOp(args) => args.run(handles.stream()),
        }
    }
}

impl ScalarOpArgs<CuFunction, DevicePtr> {
    pub unsafe fn run(&self, stream: &CudaStream) {
        let ScalarOpArgs { kernel, operands, operand_kinds: _ } = self;
        kernel.run(stream, operands);
    }
}

impl ReduceOpArgs<CuFunction, DevicePtr> {
    pub unsafe fn run(&self, stream: &CudaStream) {
        let ReduceOpArgs { kernel, input, output } = self;
        kernel.run(stream, input, output)
    }
}

impl SoftmaxOpArgs<CuFunction, DevicePtr> {
    pub unsafe fn run(&self, stream: &CudaStream) {
        let SoftmaxOpArgs { kernel, input, output } = self;
        kernel.run(stream, input, output)
    }
}

impl LayernormOpArgs<CuFunction, DevicePtr> {
    pub unsafe fn run(&self, stream: &CudaStream) {
        let LayernormOpArgs {
            kernel,
            input0,
            input1,
            output,
        } = self;
        kernel.run(stream, input0, input1.as_ref(), output)
    }
}

impl GatherOpArgs<CuFunction, DevicePtr> {
    pub unsafe fn run(&self, stream: &CudaStream) {
        let GatherOpArgs {
            kernel,
            input,
            indices,
            output,
        } = self;
        kernel.run(stream, input, indices, output)
    }
}
