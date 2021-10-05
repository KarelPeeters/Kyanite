use std::collections::HashMap;

use bytemuck::cast_slice;

use cuda_sys::bindings::{cudnnConvolutionFwdAlgo_t, cudnnOpTensorOp_t};
use cuda_sys::wrapper::descriptor::{ConvolutionDescriptor, TensorOpDescriptor};
use cuda_sys::wrapper::group::{ConvolutionArgs, TensorOpArgs};
use cuda_sys::wrapper::handle::{CudnnHandle, Device};
use cuda_sys::wrapper::mem::DeviceMem;
use nn_graph::graph::{ConvShape, Operation, Value};
use nn_graph::shape::ConcreteShape;

use crate::tensor::{len_from_shape_stride, Tensor};

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

    pub fn visit(&mut self, value: Value, shape: &ConcreteShape, operation: &Operation) {
        let result_buffer: Tensor = match operation {
            &Operation::Input { index } => {
                let buffer = self.alloc_buffer(shape.size());
                self.plan.push(Step::CopyInput { index, mem: buffer.view() });
                Tensor::new_basic(buffer, shape.clone())
            }
            Operation::Constant { data } => {
                let buffer = self.alloc_buffer(shape.size());

                // safety: we just allocated the buffer so we're the only one that can mutate it
                unsafe {
                    buffer.copy_from_host(cast_slice(&**data));
                }

                Tensor::new_basic(buffer, shape.clone())
            }
            &Operation::View { input } => {
                let input_tensor = self.get(input);

                if input_tensor.has_basic_strides {
                    // just use new basic strides based on the new shape
                    Tensor::new_basic(input_tensor.mem.view(), shape.clone())
                } else {
                    //TODO generalize this implementation, look at pytorch implementation:
                    //  https://github.com/pytorch/pytorch/blob/560cd881956bbf425251d63f0ff0f9085a759447/aten/src/ATen/TensorUtils.cpp#L335-L346

                    let mut output_strides = vec![];
                    let mut next_i = 0;

                    for &s in &shape.dims {
                        let i_s = input_tensor.shape.dims[next_i];
                        if s == i_s {
                            output_strides.push(input_tensor.strides[next_i]);
                            next_i += 1;
                        } else if s == 1 {
                            output_strides.push(0);
                        } else if i_s == 1 {
                            next_i += 1;
                        }
                    }

                    Tensor::new_special(
                        input_tensor.mem.view(),
                        shape.clone(),
                        output_strides,
                    )
                }
            }
            &Operation::Slice { input, axis, start, end: _ } => {
                let input_tensor = self.get(input);

                // to slice into a tensor:
                //  * use the new shape
                //  * keep the old strides
                //  * offset initial pointer to account for `start`
                //  * limit the buffer length based on end
                let strides = input_tensor.strides.clone();
                let offset = strides[axis] * start;
                let len = len_from_shape_stride(shape, &strides) * 4;
                let mem = input_tensor.mem.slice(offset, len);

                Tensor::new_special(mem, shape.clone(), strides)
            }
            &Operation::Conv { input, filter, conv_shape } => {
                self.visit_conv(shape, input, filter, conv_shape)
            }
            &Operation::Add { left, right, subtract } => {
                self.visit_op(shape, left, right, cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD, subtract)
            }
            &Operation::Mul { left, right } => {
                self.visit_op(shape, left, right, cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MUL, false)
            }
            &Operation::Clamp { input, min, max } => {
                let mut curr = self.get(input).view();
                if min != f32::NEG_INFINITY {
                    curr = self.visit_clamp(shape, &curr, min, cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MAX);
                }
                if max != f32::INFINITY {
                    curr = self.visit_clamp(shape, &curr, max, cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MIN);
                }
                curr
            }
        };

        let prev = self.map.insert(value, result_buffer);
        assert!(prev.is_none());
    }

    pub fn visit_output(&mut self, index: usize, value: Value) {
        let value = self.get(value);
        let mem = value.mem.view();
        self.plan.push(Step::CopyOutput { index, mem })
    }

    fn visit_conv(&mut self, output_shape: &ConcreteShape, input: Value, filter: Value, conv_shape: ConvShape) -> Tensor {
        let conv_desc = ConvolutionDescriptor::new(
            conv_shape.padding as i32, conv_shape.padding as i32,
            1, 1,
            1, 1,
        );

        let output = Tensor::new_basic(
            self.alloc_buffer(output_shape.size()),
            output_shape.clone(),
        );

        let input_desc = self.get(input).descriptor();
        let output_desc = output.descriptor();
        let filter_desc = self.get(filter).descriptor_filter();

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

        self.plan.push(Step::Conv { args });
        output
    }

    fn visit_op(&mut self, output_shape: &ConcreteShape, left: Value, right: Value, op: cudnnOpTensorOp_t, negate_right: bool) -> Tensor {
        let op_desc = TensorOpDescriptor::new(op);
        let alpha_2 = if negate_right { -1.0 } else { 1.0 };

        let output = Tensor::new_basic(
            self.alloc_buffer(output_shape.size()),
            output_shape.clone(),
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

    fn visit_clamp(&mut self, output_shape: &ConcreteShape, input: &Tensor, limit: f32, op: cudnnOpTensorOp_t) -> Tensor {
        assert!(op == cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MIN || op == cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MAX);
        let op_desc = TensorOpDescriptor::new(op);

        // create a one-element tensor to hold the limit value
        let right = Tensor::new_basic(
            self.alloc_buffer(1),
            ConcreteShape::new(vec![1; output_shape.rank()]),
        );

        // safety: we just allocated this memory so no one else can modify it
        unsafe {
            right.mem.copy_from_host(cast_slice(&[limit]));
        }

        let output = Tensor::new_basic(
            self.alloc_buffer(output_shape.size()),
            output_shape.clone(),
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
    Conv { args: ConvolutionArgs },
    TensorOp { args: TensorOpArgs },
    CopyOutput { index: usize, mem: DeviceMem },
}
