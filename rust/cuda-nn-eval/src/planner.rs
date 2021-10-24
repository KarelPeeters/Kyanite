use std::collections::HashMap;

use bytemuck::cast_slice;

use cuda_sys::bindings::{cudnnConvolutionFwdAlgo_t, cudnnOpTensorOp_t};
use cuda_sys::wrapper::descriptor::{ConvolutionDescriptor, TensorOpDescriptor};
use cuda_sys::wrapper::group::{ConvolutionArgs, TensorOpArgs};
use cuda_sys::wrapper::handle::{CudnnHandle, Device};
use cuda_sys::wrapper::mem::device::DeviceMem;
use nn_graph::graph::{ConvShape, Operation, Value};
use nn_graph::shape::ConcreteShape;

use crate::shape::StridedShape;
use crate::tensor::Tensor;

pub struct Planner {
    handle: CudnnHandle,
    device: Device,
    map: HashMap<Value, Tensor>,
    plan: Vec<Step>,
}

//TODO switch back to automatic profiling, even though this algo seems to always be (close to) the best one
const CONV_ALGO: cudnnConvolutionFwdAlgo_t = cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

impl Planner {
    pub fn new(handle: CudnnHandle) -> Self {
        Planner {
            device: handle.device(),
            handle,
            plan: vec![],
            map: Default::default(),
        }
    }

    pub fn finish(self) -> (CudnnHandle, Vec<Step>) {
        (self.handle, self.plan)
    }

    fn alloc_buffer(&mut self, size: usize) -> DeviceMem {
        DeviceMem::alloc(size * 4, self.device)
    }

    fn get(&self, value: Value) -> &Tensor {
        self.map.get(&value).unwrap()
    }

    pub fn visit(&mut self, value: Value, output_shape: ConcreteShape, operation: &Operation) {
        let result_buffer: Tensor = match operation {
            &Operation::Input { index } => {
                let buffer = self.alloc_buffer(output_shape.size());
                self.plan.push(Step::CopyInput { index, mem: buffer.view() });
                Tensor::new(buffer, StridedShape::new_simple(output_shape.dims))
            }
            Operation::Constant { data } => {
                let buffer = self.alloc_buffer(output_shape.size());

                // safety: we just allocated the buffer so we're the only one that can mutate it
                unsafe {
                    buffer.copy_from_host(cast_slice(&**data));
                }

                Tensor::new(buffer, StridedShape::new_simple(output_shape.dims))
            }
            &Operation::View { input } => {
                let input_tensor = self.get(input);

                let new_shape = input_tensor.shape.view(output_shape.dims.clone())
                    .unwrap_or_else(|| panic!("Cannot view shape {:?} as {:?}", input_tensor.shape, output_shape));

                Tensor::new(input_tensor.mem.view(), new_shape)
            }
            &Operation::Slice { input, axis, start, end } => {
                // Steps to slice a tensor:
                //  * use the new shape
                //  * keep the old strides
                //  * offset initial pointer to account for `start`
                //  * limit the buffer length based on the new size

                let input_tensor = self.get(input);
                let result_shape = input_tensor.shape.slice(axis, start, end);

                let start_bytes = result_shape.strides()[axis] * start * 4;
                let len_bytes = result_shape.strided_size() * 4;

                let mem = input_tensor.mem.slice(start_bytes, len_bytes);
                Tensor::new(mem, result_shape)
            }
            &Operation::Conv { input, filter, conv_shape } => {
                self.visit_conv(output_shape, input, filter, conv_shape)
            }
            &Operation::Add { left, right, subtract } => {
                self.visit_op(output_shape, left, right, cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD, subtract)
            }
            &Operation::Mul { left, right } => {
                self.visit_op(output_shape, left, right, cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MUL, false)
            }
            &Operation::Clamp { input, min, max } => {
                let mut curr = self.get(input).view();
                if min != f32::NEG_INFINITY {
                    curr = self.visit_clamp(output_shape.clone(), &curr, min, cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MAX);
                }
                if max != f32::INFINITY {
                    curr = self.visit_clamp(output_shape, &curr, max, cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MIN);
                }
                curr
            }
        };

        let prev = self.map.insert(value, result_buffer);
        assert!(prev.is_none());
    }

    pub fn visit_output(&mut self, index: usize, value: Value) -> Tensor {
        let tensor = self.get(value).view();
        self.plan.push(Step::CopyOutput { index, tensor: tensor.view() });
        tensor
    }

    fn visit_conv(&mut self, output_shape: ConcreteShape, input: Value, filter: Value, conv_shape: ConvShape) -> Tensor {
        let conv_desc = ConvolutionDescriptor::new(
            conv_shape.padding as i32, conv_shape.padding as i32,
            1, 1,
            1, 1,
        );

        let output = Tensor::new(
            self.alloc_buffer(output_shape.size()),
            StridedShape::new_simple(output_shape.dims),
        );

        let input_desc = self.get(input).descriptor();
        let output_desc = output.descriptor();
        let filter_desc = self.get(filter).filter_descriptor();

        let algo = CONV_ALGO;
        let work_size_bytes = conv_desc.workspace_size(&mut self.handle, algo, &input_desc, &filter_desc, &output_desc);
        let work_mem = DeviceMem::alloc(work_size_bytes, self.device);

        let input = self.get(input);
        let filter = self.get(filter);

        let args = ConvolutionArgs {
            conv_desc,
            algo,
            work_mem,
            filter_desc,
            filter_mem: filter.mem.view(),
            input_desc,
            input_mem: input.mem.view(),
            output_desc,
            output_mem: output.mem.view(),
        };

        self.plan.push(Step::Conv { shape: conv_shape, args });
        output
    }

    fn visit_op(&mut self, output_shape: ConcreteShape, left: Value, right: Value, op: cudnnOpTensorOp_t, negate_right: bool) -> Tensor {
        let op_desc = TensorOpDescriptor::new(op);
        let alpha_2 = if negate_right { -1.0 } else { 1.0 };

        let output = Tensor::new(
            self.alloc_buffer(output_shape.size()),
            StridedShape::new_simple(output_shape.dims),
        );

        let left = self.get(left);
        let right = self.get(right);

        // TODO they're not 4d per se, but we need to create 4D descriptors for them anyway

        let args = TensorOpArgs {
            op_desc,
            alpha_1: 1.0,
            input_1_desc: left.descriptor(),
            input_1_mem: left.mem.view(),
            alpha_2,
            input_2_desc: right.descriptor(),
            input_2_mem: right.mem.view(),
            beta: 0.0,
            output_desc: output.descriptor(),
            output_mem: output.mem.view(),
        };

        self.plan.push(Step::TensorOp { args });
        output
    }

    fn visit_clamp(&mut self, output_shape: ConcreteShape, input: &Tensor, limit: f32, op: cudnnOpTensorOp_t) -> Tensor {
        assert!(op == cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MIN || op == cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MAX);
        let op_desc = TensorOpDescriptor::new(op);

        // create a one-element tensor with the same rank as `left` to hold the limit value
        let right = Tensor::new(
            self.alloc_buffer(1),
            StridedShape::new_simple(vec![1; output_shape.rank()]),
        );

        // safety: we just allocated this memory so no one else can modify it
        unsafe {
            right.mem.copy_from_host(cast_slice(&[limit]));
        }

        let output = Tensor::new(
            self.alloc_buffer(output_shape.size()),
            StridedShape::new_simple(output_shape.dims).clone(),
        );

        let args = TensorOpArgs {
            op_desc,
            alpha_1: 1.0,
            input_1_desc: input.descriptor(),
            input_1_mem: input.mem.view(),
            alpha_2: 1.0,
            input_2_desc: right.descriptor(),
            input_2_mem: right.mem.view(),
            beta: 0.0,
            output_desc: output.descriptor(),
            output_mem: output.mem.view(),
        };

        self.plan.push(Step::TensorOp { args });
        output
    }
}

#[derive(Debug)]
pub enum Step {
    CopyInput { index: usize, mem: DeviceMem },
    Conv { shape: ConvShape, args: ConvolutionArgs },
    TensorOp { args: TensorOpArgs },
    CopyOutput { index: usize, tensor: Tensor },
}
