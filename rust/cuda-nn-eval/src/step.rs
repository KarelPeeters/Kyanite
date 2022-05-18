use std::ops::ControlFlow;

use internal_iterator::InternalIterator;

use crate::autokernel::layernorm::LayernormKernel;
use cuda_sys::wrapper::group::{BatchedMatMulArgs, FusedConvolutionArgs};

use crate::autokernel::reduce::ReduceKernel;
use crate::autokernel::scalar::ScalarKernel;
use crate::autokernel::softmax::SoftmaxKernel;
use crate::offset_tensor::PtrTensor;

#[derive(Debug)]
pub struct StepInfo<P> {
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
    Gather(GatherArgs<P>),
}

#[derive(Debug)]
pub struct GatherArgs<P> {
    pub input: PtrTensor<P>,
    pub axis: usize,
    pub indices: PtrTensor<P>,
    pub output: PtrTensor<P>,
}

#[derive(Debug)]
pub struct ScalarOpArgs<P> {
    pub kernel: ScalarKernel,
    pub operands: Vec<PtrTensor<P>>,
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
pub struct LayernormOpArgs<P> {
    pub kernel: LayernormKernel,
    pub input: PtrTensor<P>,
    pub output: PtrTensor<P>,
}

#[derive(Debug)]
pub struct PlanStepOperands<'a, P>(&'a Step<P>);

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
                input: args.input.map_ptr(&mut f),
                output: args.output.map_ptr(&mut f),
            }),
            Step::Gather(args) => Step::Gather(GatherArgs {
                input: args.input.map_ptr(&mut f),
                axis: args.axis,
                indices: args.indices.map_ptr(&mut f),
                output: args.output.map_ptr(&mut f),
            }),
        }
    }

    pub fn for_each_ptr<'a, R>(&'a self, mut f: impl FnMut(&'a P) -> ControlFlow<R>) -> ControlFlow<R> {
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
                f(work_ptr)?;
                f(filter_ptr)?;
                f(input_ptr)?;
                f(bias_ptr)?;
                if let Some(res_ptr) = res_ptr {
                    f(res_ptr)?;
                }
                f(output_ptr)?;
            }
            Step::MatMul(BatchedMatMulArgs {
                m: _,
                n: _,
                k: _,
                alpha: _,
                beta: _,
                a,
                b,
                c,
                batch_count: _,
            }) => {
                f(&a.ptr)?;
                f(&b.ptr)?;
                f(&c.ptr)?;
            }
            Step::ScalarOp(ScalarOpArgs { kernel: _, operands }) => operands.iter().map(|a| a.ptr()).try_for_each(f)?,
            Step::ReduceOp(ReduceOpArgs {
                kernel: _,
                input,
                output,
            }) => {
                f(input.ptr())?;
                f(output.ptr())?;
            }
            Step::SoftmaxOp(SoftmaxOpArgs {
                kernel: _,
                input,
                output,
            }) => {
                f(input.ptr())?;
                f(output.ptr())?;
            }
            Step::LayernormOp(LayernormOpArgs {
                kernel: _,
                input,
                output,
            }) => {
                f(input.ptr())?;
                f(output.ptr())?;
            }
            Step::Gather(GatherArgs {
                input,
                axis: _,
                indices,
                output,
            }) => {
                f(input.ptr())?;
                f(indices.ptr())?;
                f(output.ptr())?;
            }
        }

        ControlFlow::Continue(())
    }
}

impl<'a, P> InternalIterator for PlanStepOperands<'a, P> {
    type Item = &'a P;

    fn try_for_each<R, F>(self, f: F) -> ControlFlow<R>
    where
        F: FnMut(Self::Item) -> ControlFlow<R>,
    {
        self.0.for_each_ptr(f)
    }
}
