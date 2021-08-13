use std::collections::HashMap;

use bytemuck::{cast_slice, cast_slice_mut};
use unwrap_match::unwrap_match;

use cuda_sys::bindings::{cudnnConvolutionFwdAlgo_t, cudnnDataType_t, cudnnTensorFormat_t, cudnnActivationMode_t};
use cuda_sys::wrapper::descriptor::{ActivationDescriptor, ConvolutionDescriptor, FilterDescriptor, TensorDescriptor};
use cuda_sys::wrapper::handle::CudnnHandle;
use cuda_sys::wrapper::mem::DeviceMem;
use cuda_sys::wrapper::operation::{ResInput, run_conv_bias_res_activation};

use crate::fuser::{FusedGraph, FusedValue, FusedValueInfo, Activation};
use crate::graph::{ConvShape, Graph, Operation};

#[derive(Debug)]
pub struct CudaGraphExecutor {
    handle: CudnnHandle,

    plan: Vec<PlannedOperation>,
    buffers: Vec<DeviceMem>,

    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

#[derive(Debug)]
struct PlannedOperation {
    input_index: usize,
    filter_index: usize,
    bias_index: usize,
    res_index: Option<usize>,
    output_index: usize,

    input_desc: TensorDescriptor,
    conv_desc: ConvolutionDescriptor,
    filter_desc: FilterDescriptor,
    bias_desc: TensorDescriptor,
    act_desc: ActivationDescriptor,
    output_desc: TensorDescriptor,

    algo: cudnnConvolutionFwdAlgo_t,
    workspace_mem: DeviceMem,
}

struct OperationBuffers<'a> {
    input: &'a DeviceMem,
    filter: &'a DeviceMem,
    bias: &'a DeviceMem,
    res: ResInput<'a>,
    output: &'a mut DeviceMem,
}

impl PlannedOperation {
    fn get_buffers<'a>(&self, buffers: &'a mut Vec<DeviceMem>) -> OperationBuffers<'a> {
        //TODO this implements a very conservative requirement, it's not clear if eg the filter and input can be the same mem
        //  it should be fine for now with how the rest of the planning pipeline works though
        let mut indices = vec![
            self.input_index, self.filter_index, self.bias_index,
            self.res_index.unwrap_or(usize::MAX),
            self.output_index,
        ];
        let len_before = indices.len();
        indices.dedup();
        assert_eq!(indices.len(), len_before, "Found duplicate index in planned operation");

        // safe because we just checked that all indices are distinct
        unsafe {
            let res = match self.res_index {
                None => ResInput::Zero,
                Some(res_index) => ResInput::Other(&*(&buffers[res_index] as *const _))
            };

            OperationBuffers {
                input: &*(&buffers[self.input_index] as *const _),
                filter: &*(&buffers[self.filter_index] as *const _),
                bias: &*(&buffers[self.bias_index] as *const _),
                res,
                output: &mut *(&mut buffers[self.output_index] as *mut _),
            }
        }
    }
}

//TODO switch back to automatic profiling, even though this algo seems to always be (close to) the best one
const CONV_ALGO: cudnnConvolutionFwdAlgo_t = cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

impl CudaGraphExecutor {
    //TODO implement memory reuse, with some "register" allocation scheme
    pub fn new(mut handle: CudnnHandle, graph: &Graph) -> Self {
        let device = handle.device();
        let fused_graph = FusedGraph::new(graph);

        let mut buffer_map: HashMap<FusedValue, usize> = Default::default();

        let mut plan = vec![];
        let mut buffers = vec![];
        let mut inputs = vec![];

        for fused_value in fused_graph.schedule() {
            let value = fused_graph[fused_value].value();
            let value_info = &graph[value];
            let shape = value_info.shape;

            let output_index = buffers.len();
            let output_mem = DeviceMem::alloc(shape.iter().product::<i32>() as usize * 4, device);
            buffers.push(output_mem);
            buffer_map.insert(fused_value, output_index);

            match fused_graph[fused_value] {
                FusedValueInfo::Input(_) => {
                    // register as input
                    inputs.push(output_index);
                }
                FusedValueInfo::Constant(_) => {
                    // copy constant data to device mem
                    let data = unwrap_match!(&value_info.operation, Operation::Constant { data } => &**data);
                    buffers[output_index].copy_from_host(cast_slice(data));
                }
                FusedValueInfo::FusedOperation {
                    value: _,
                    input, input_shape_view, res_input, bias, filter,
                    conv_shape, act_mode
                } => {
                    let ConvShape {
                        batch_size: _,
                        input_channels, output_channels,
                        input_size: _, kernel_size, padding, output_size: _
                    } = conv_shape;

                    let input_desc = shape_to_tensor_desc(input_shape_view);
                    let output_desc = shape_to_tensor_desc(graph[value].shape);
                    let bias_desc = shape_to_tensor_desc(graph[fused_graph[bias].value()].shape);

                    let filter_desc = FilterDescriptor::new(
                        output_channels, input_channels, kernel_size, kernel_size,
                        cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                    );

                    let conv_desc = ConvolutionDescriptor::new(
                        padding, padding, 1, 1, 1, 1, cudnnDataType_t::CUDNN_DATA_FLOAT,
                    );

                    let act_mode = match act_mode {
                        Activation::Linear => cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY,
                        Activation::Relu => cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
                    };

                    let workspace_size = conv_desc.workspace_size(
                        &mut handle,
                        CONV_ALGO,
                        &input_desc, &filter_desc, &output_desc,
                    );

                    let workspace_mem = DeviceMem::alloc(workspace_size, device);

                    let operation = PlannedOperation {
                        input_index: *buffer_map.get(&input).unwrap(),
                        filter_index: *buffer_map.get(&filter).unwrap(),
                        bias_index: *buffer_map.get(&bias).unwrap(),
                        res_index: res_input.map(|res_index| *buffer_map.get(&res_index).unwrap()),
                        output_index,
                        input_desc,
                        conv_desc,
                        filter_desc,
                        bias_desc,
                        act_desc: ActivationDescriptor::new(act_mode, 0.0),
                        output_desc,
                        algo: CONV_ALGO,
                        workspace_mem,
                    };
                    plan.push(operation);
                }
            }
        }

        let outputs = graph.outputs().iter().map(|&value| {
            // find buffer index for the fused value corresponding to this value
            let fused_value =fused_graph.find(value);
            *buffer_map.get(&fused_value)
                .unwrap_or_else(|| panic!("Output {:?} not found in buffer map", value))
        }).collect();

        CudaGraphExecutor { handle, buffers, plan, inputs, outputs }
    }

    pub fn run(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) {
        assert_eq!(self.inputs.len(), inputs.len(), "Wrong number of inputs");
        assert_eq!(self.outputs.len(), outputs.len(), "Wrong number of outputs");

        unsafe {
            // copy inputs to buffers
            for i in 0..inputs.len() {
                self.buffers[self.inputs[i]]
                    .copy_from_host_async(cast_slice(inputs[i]), self.handle.stream());
            }

            // run operations
            for i in 0..self.plan.len() {
                self.run_op(i)
            }

            // copy outputs back
            for i in 0..outputs.len() {
                self.buffers[self.outputs[i]]
                    .copy_to_host_async(cast_slice_mut(outputs[i]), self.handle.stream());
            }

            self.handle.stream().synchronize();
        }
    }

    fn run_op(&mut self, op_index: usize) {
        let op = &mut self.plan[op_index];
        let buffers = op.get_buffers(&mut self.buffers);

        run_conv_bias_res_activation(
            &mut self.handle,
            &op.act_desc,
            &op.conv_desc,
            op.algo,
            &mut op.workspace_mem,
            &op.filter_desc,
            buffers.filter,
            &op.input_desc,
            buffers.input,
            buffers.res,
            &op.bias_desc,
            buffers.bias,
            &op.output_desc,
            buffers.output,
        )
    }
}

fn shape_to_tensor_desc(shape: [i32; 4]) -> TensorDescriptor {
    TensorDescriptor::new(
        shape[0], shape[1], shape[2], shape[3],
        cudnnDataType_t::CUDNN_DATA_FLOAT,
        cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
    )
}