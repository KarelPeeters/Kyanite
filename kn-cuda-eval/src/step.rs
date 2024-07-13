use std::ops::ControlFlow;

use internal_iterator::InternalIterator;
use itertools::zip_eq;

use kn_cuda_sys::wrapper::group::{BatchedMatMulArgs, FusedConvolutionArgs};
use kn_cuda_sys::wrapper::handle::{CublasHandle, CudaDevice, CudaStream, CudnnHandle};
use kn_cuda_sys::wrapper::mem::device::DevicePtr;
use kn_graph::graph::Value;

use crate::autokernel::gather::GatherKernel;
use crate::autokernel::layernorm::LayernormKernel;
use crate::autokernel::reduce::ReduceKernel;
use crate::autokernel::scalar::ScalarKernel;
use crate::autokernel::softmax::SoftmaxKernel;
use crate::offset_tensor::PtrTensor;

#[derive(Debug)]
pub struct StepInfo<P> {
    pub debug_value: Value,
    pub debug_id: String,
    pub step: Step<P>,
}

#[derive(Debug)]
pub enum Step<P> {
    Conv(FusedConvolutionArgs<P>),
    MatMul(BatchedMatMulArgs<P>),
    ScalarOp(ScalarOpArgs<P>),
    ReduceOp(ReduceOpArgs<P>),
    SoftmaxOp(SoftmaxOpArgs<P>),
    LayernormOp(LayernormOpArgs<P>),
    GatherOp(GatherOpArgs<P>),
}

#[derive(Debug)]
pub struct ScalarOpArgs<P> {
    pub kernel: ScalarKernel,
    pub operands: Vec<PtrTensor<P>>,
    pub operand_kinds: Vec<OperandKind>,
}

#[derive(Debug)]
pub struct ReduceOpArgs<P> {
    pub kernel: ReduceKernel,
    pub input: PtrTensor<P>,
    pub output: PtrTensor<P>,
}

#[derive(Debug)]
pub struct SoftmaxOpArgs<P> {
    pub kernel: SoftmaxKernel,
    pub input: PtrTensor<P>,
    pub output: PtrTensor<P>,
}

#[derive(Debug)]
pub struct GatherOpArgs<P> {
    pub kernel: GatherKernel,
    pub input: PtrTensor<P>,
    pub indices: PtrTensor<P>,
    pub output: PtrTensor<P>,
}

#[derive(Debug)]
pub struct LayernormOpArgs<P> {
    pub kernel: LayernormKernel,
    pub input0: PtrTensor<P>,
    pub input1: Option<PtrTensor<P>>,
    pub output: PtrTensor<P>,
}

#[derive(Debug)]
pub struct PlanStepOperands<'a, P>(&'a Step<P>);

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
impl<P> Step<P> {
    pub fn ptr_operands(&self) -> PlanStepOperands<P> {
        PlanStepOperands(self)
    }

    pub fn map_ptrs<K>(self, mut f: impl FnMut(P) -> K) -> Step<K> {
        match self {
            Step::Conv(args) => Step::Conv(FusedConvolutionArgs {
                conv_desc: args.conv_desc,
                algo: args.algo,
                work_ptr: f(args.work_ptr),
                work_size_bytes: args.work_size_bytes,
                filter_desc: args.filter_desc,
                filter_ptr: f(args.filter_ptr),
                input_desc: args.input_desc,
                input_ptr: f(args.input_ptr),
                res_ptr: args.res_ptr.map(&mut f),
                bias_desc: args.bias_desc,
                bias_ptr: f(args.bias_ptr),
                act_desc: args.act_desc,
                output_desc: args.output_desc,
                output_ptr: f(args.output_ptr),
            }),
            Step::MatMul(args) => Step::MatMul(BatchedMatMulArgs {
                m: args.m,
                n: args.n,
                k: args.k,
                alpha: args.alpha,
                beta: args.beta,
                a: args.a.map_ptr(&mut f),
                b: args.b.map_ptr(&mut f),
                c: args.c.map_ptr(&mut f),
                batch_count: args.batch_count,
            }),
            Step::ScalarOp(args) => Step::ScalarOp(ScalarOpArgs {
                kernel: args.kernel,
                operands: args.operands.into_iter().map(|op| op.map_ptr(&mut f)).collect(),
                operand_kinds: args.operand_kinds.clone(),
            }),
            Step::ReduceOp(args) => Step::ReduceOp(ReduceOpArgs {
                kernel: args.kernel,
                input: args.input.map_ptr(&mut f),
                output: args.output.map_ptr(&mut f),
            }),
            Step::SoftmaxOp(args) => Step::SoftmaxOp(SoftmaxOpArgs {
                kernel: args.kernel,
                input: args.input.map_ptr(&mut f),
                output: args.output.map_ptr(&mut f),
            }),
            Step::LayernormOp(args) => Step::LayernormOp(LayernormOpArgs {
                kernel: args.kernel,
                input0: args.input0.map_ptr(&mut f),
                input1: args.input1.map(|op| op.map_ptr(&mut f)),
                output: args.output.map_ptr(&mut f),
            }),
            Step::GatherOp(args) => Step::GatherOp(GatherOpArgs {
                kernel: args.kernel,
                input: args.input.map_ptr(&mut f),
                indices: args.indices.map_ptr(&mut f),
                output: args.output.map_ptr(&mut f),
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

impl<'a, P> InternalIterator for PlanStepOperands<'a, P> {
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

    #[allow(dead_code)]
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
    pub fn device(&self) -> CudaDevice {
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

impl Step<DevicePtr> {
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

impl ScalarOpArgs<DevicePtr> {
    pub unsafe fn run(&self, stream: &CudaStream) {
        let ScalarOpArgs { kernel, operands, operand_kinds: _ } = self;
        kernel.run(stream, operands);
    }
}

impl ReduceOpArgs<DevicePtr> {
    pub unsafe fn run(&self, stream: &CudaStream) {
        let ReduceOpArgs { kernel, input, output } = self;
        kernel.run(stream, input, output)
    }
}

impl SoftmaxOpArgs<DevicePtr> {
    pub unsafe fn run(&self, stream: &CudaStream) {
        let SoftmaxOpArgs { kernel, input, output } = self;
        kernel.run(stream, input, output)
    }
}

impl LayernormOpArgs<DevicePtr> {
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

impl GatherOpArgs<DevicePtr> {
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
