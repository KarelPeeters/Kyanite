use std::ops::ControlFlow;

use internal_iterator::InternalIterator;

use cuda_sys::wrapper::group::{BatchedMatMulArgs, FusedConvolutionArgs, MatMulOperand, TensorOpArgs};

use crate::offset_tensor::PtrTensor;

#[derive(Debug)]
pub enum Step<P> {
    Conv(FusedConvolutionArgs<P>),
    MatMul(BatchedMatMulArgs<P>),
    TensorOp(TensorOpArgs<P>),
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
                a: matmul_args_map_ptr(args.a, &mut f),
                b: matmul_args_map_ptr(args.b, &mut f),
                c: matmul_args_map_ptr(args.c, &mut f),
                batch_count: args.batch_count,
            }),
            Step::TensorOp(args) => Step::TensorOp(TensorOpArgs {
                op_desc: args.op_desc,
                alpha_1: args.alpha_1,
                input_1_desc: args.input_1_desc,
                input_1_ptr: f(args.input_1_ptr),
                alpha_2: args.alpha_2,
                input_2_desc: args.input_2_desc,
                input_2_ptr: f(args.input_2_ptr),
                beta: args.beta,
                output_desc: args.output_desc,
                output_ptr: f(args.output_ptr),
            }),
            Step::Gather(args) => Step::Gather(GatherArgs {
                input: args.input.map_inner(&mut f),
                axis: args.axis,
                indices: args.indices.map_inner(&mut f),
                output: args.output.map_inner(&mut f),
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
            Step::TensorOp(TensorOpArgs {
                op_desc: _,
                alpha_1: _,
                input_1_desc: _,
                input_1_ptr,
                alpha_2: _,
                input_2_desc: _,
                input_2_ptr,
                beta: _,
                output_desc: _,
                output_ptr,
            }) => {
                f(input_1_ptr)?;
                f(input_2_ptr)?;
                f(output_ptr)?;
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

fn matmul_args_map_ptr<P, K>(operand: MatMulOperand<P>, mut f: impl FnMut(P) -> K) -> MatMulOperand<K> {
    MatMulOperand {
        ptr: f(operand.ptr),
        trans: operand.trans,
        ld: operand.ld,
        stride: operand.stride,
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
