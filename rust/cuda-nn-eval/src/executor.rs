use std::collections::HashMap;

use bytemuck::{cast_slice, cast_slice_mut};

use cuda_sys::bindings::{cudnnConvolutionFwdAlgo_t, cudnnDataType_t, cudnnTensorFormat_t};
use cuda_sys::wrapper::descriptor::{ActivationDescriptor, ConvolutionDescriptor};
use cuda_sys::wrapper::group::Tensor;
use cuda_sys::wrapper::handle::CudnnHandle;
use cuda_sys::wrapper::mem::DeviceMem;
use cuda_sys::wrapper::operation::{ResInput, run_conv_bias_res_activation};

use crate::graph::Graph;
use crate::load::GraphParams;
use crate::planner::{FusedGraph, FusedValueInfo};

#[derive(Debug)]
pub struct CudaGraphExecutor {
    handle: CudnnHandle,

    params: GraphParams,
    plan: Vec<PlannedOperation>,

    buffers: Vec<Tensor>,

    inputs: Vec<usize>,
    outputs: Vec<usize>,
}

#[derive(Debug)]
struct PlannedOperation {
    indices: BufferIndices,

    filter_index: usize,
    bias_index: usize,

    act: ActivationDescriptor,
    conv: ConvolutionDescriptor,
    algo: cudnnConvolutionFwdAlgo_t,

    workspace: DeviceMem,
}

#[derive(Debug, Copy, Clone)]
struct BufferIndices {
    input_index: usize,
    res_input_index: Option<usize>,
    output_index: usize,
}

#[derive(Debug)]
struct Buffers<'a> {
    input: &'a Tensor,
    res_input: ResInput<'a>,
    output: &'a mut Tensor,
}

impl BufferIndices {
    /// Get references to the input, output and res buffers.
    /// This function asserts that they don't overlap where not allowed so it's safe.
    fn get_buffers(self, buffers: &mut Vec<Tensor>) -> Buffers {
        let BufferIndices { input_index: input_mem, output_index: output_mem, res_input_index: res } = self;

        assert_ne!(input_mem, output_mem);
        assert_ne!(Some(input_mem), res);

        unsafe {
            let res_type = match res {
                None => ResInput::Zero,
                Some(res_mem) => {
                    if res_mem == output_mem {
                        ResInput::Output
                    } else {
                        let tensor = &*(&buffers[res_mem] as *const Tensor);
                        ResInput::Other { desc: &tensor.desc, mem: &tensor.mem }
                    }
                }
            };

            Buffers {
                input: &*(&buffers[input_mem] as *const _),
                output: &mut *(&mut buffers[output_mem] as *mut _),
                res_input: res_type,
            }
        }
    }
}

//TODO switch back to automatic profiling, even though this algo seems to always be (close to) the best one
const CONV_ALGO: cudnnConvolutionFwdAlgo_t = cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

impl CudaGraphExecutor {
    //TODO implement memory reuse, with some "register" allocation scheme
    pub fn new(mut handle: CudnnHandle, graph: &Graph, params: GraphParams) -> Self {
        let fused_graph = FusedGraph::new(graph);

        let mut plan = vec![];
        let mut buffers = vec![];
        let mut buffer_map = HashMap::new();
        let mut inputs = vec![];
        let mut outputs = vec![];

        for fused_value in fused_graph.schedule() {
            let fused_info = fused_graph[fused_value];
            let value = fused_info.value();

            // allocate "output" buffer
            let shape = graph[value].shape;
            let output_buffer = Tensor::new(
                shape[0], shape[1], shape[2], shape[3],
                cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, handle.device(),
            );
            let output_index = buffers.len();
            buffers.push(output_buffer);
            assert!(buffer_map.insert(fused_value, output_index).is_none());

            // register buffer as output
            if graph.outputs().contains(&value) {
                outputs.push(output_index);
            }

            match fused_info {
                FusedValueInfo::Input(_) => {
                    // register buffer as input
                    inputs.push(output_index)
                }
                FusedValueInfo::FusedOperation {
                    value: _,
                    input_fused, res_input,
                    output_channels, kernel_size, padding,
                    filter_index, bias_index,
                    act_mode
                } => {
                    //check parameter sizes just to make sure
                    let input_channels = graph[fused_graph[input_fused].value()].shape[1];
                    assert_eq!(
                        [output_channels, input_channels, kernel_size, kernel_size],
                        params.filters[filter_index].desc.shape()
                    );
                    assert_eq!(
                        [1, output_channels, 1, 1],
                        params.biases[bias_index].desc.shape(),
                    );

                    // map everything to planned operation
                    let indices = BufferIndices {
                        input_index: *buffer_map.get(&input_fused).unwrap(),
                        res_input_index: res_input.map(|res| *buffer_map.get(&res).unwrap()),
                        output_index,
                    };

                    let conv = ConvolutionDescriptor::new(
                        padding, padding,
                        1, 1, 1, 1,
                        cudnnDataType_t::CUDNN_DATA_FLOAT,
                    );

                    let act = ActivationDescriptor::new(act_mode, 0.0);
                    let algo = CONV_ALGO;

                    let workspace_size = conv.workspace_size(
                        &mut handle,
                        algo,
                        &buffers[indices.input_index].desc,
                        &params.filters[filter_index].desc,
                        &buffers[indices.output_index].desc,
                    );
                    let workspace = DeviceMem::alloc(workspace_size, handle.device());

                    // figure out inputs
                    let operation = PlannedOperation {
                        indices,
                        filter_index,
                        bias_index,
                        act,
                        conv,
                        algo,
                        workspace,
                    };
                    plan.push(operation);
                }
            }
        }

        CudaGraphExecutor {
            handle,
            params,
            plan,
            buffers,
            inputs,
            outputs,
        }
    }

    pub fn eval(&mut self, inputs: &[&[f32]], outputs: &mut [&mut [f32]]) {
        assert_eq!(self.inputs.len(), inputs.len(), "Wrong number of inputs");
        assert_eq!(self.outputs.len(), outputs.len(), "Wrong number of outputs");

        for i in 0..inputs.len() {
            self.buffers[self.inputs[i]].mem.copy_from_host(cast_slice(inputs[i]));
        }

        for planned_op in &mut self.plan {
            let PlannedOperation {
                indices, filter_index, bias_index,
                act, conv, algo,
                workspace
            } = planned_op;

            let filter = &self.params.filters[*filter_index];
            let bias = &self.params.biases[*bias_index];

            let Buffers {
                res_input, input, output
            } = indices.get_buffers(&mut self.buffers);

            run_conv_bias_res_activation(
                &mut self.handle,
                act,
                conv,
                *algo,
                workspace,
                &filter.desc,
                &filter.mem,
                &input.desc,
                &input.mem,
                res_input,
                &bias.desc,
                &bias.mem,
                &output.desc,
                &mut output.mem,
            )
        }

        for i in 0..outputs.len() {
            self.buffers[self.outputs[i]].mem.copy_to_host(cast_slice_mut(outputs[i]));
        }
    }
}
