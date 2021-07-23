use bytemuck::{cast_slice, cast_slice_mut};
use itertools::Itertools;
use rand::{Rng, thread_rng};

use cuda_sys::bindings::{cudnnActivationMode_t, cudnnConvolutionFwdAlgo_t, cudnnDataType_t, cudnnTensorFormat_t};
use cuda_sys::wrapper::descriptor::{ActivationDescriptor, ConvolutionDescriptor, FilterDescriptor, TensorDescriptor};
use cuda_sys::wrapper::handle::CudnnHandle;
use cuda_sys::wrapper::mem::DeviceMem;
use cuda_sys::wrapper::operation::{find_conv_algorithms, ResType, run_conv_bias_res_activation};

//TODO initial convolution
//TODO value head
//TODO policy head

#[derive(Copy, Clone)]
pub struct NetDefinition {
    pub tower_depth: usize,
    pub tower_channels: i32,
}

struct ConvWeights {
    filter_mem: DeviceMem,
    bias_mem: DeviceMem,
}

impl ConvWeights {
    fn random(handle: &CudnnHandle, filter_desc: &FilterDescriptor, bias_desc: &TensorDescriptor) -> Self {
        let mut rng = thread_rng();

        let mut filter_mem = DeviceMem::alloc(filter_desc.size(), handle.device());
        let mut bias_mem = DeviceMem::alloc(bias_desc.size(), handle.device());

        filter_mem.copy_from_host(
            cast_slice(
                &(0..filter_mem.size() / 4)
                    .map(|_| rng.gen::<f32>())
                    .collect_vec()
            )
        );
        bias_mem.copy_from_host(
            cast_slice(
                &(0..bias_mem.size() / 4)
                    .map(|_| rng.gen::<f32>())
                    .collect_vec()
            )
        );

        ConvWeights { filter_mem, bias_mem }
    }
}

pub struct NetEvaluator {
    handle: CudnnHandle,
    batch_size: i32,

    def: NetDefinition,
    layers: Vec<(ConvWeights, ConvWeights)>,

    image_desc: TensorDescriptor,

    conv_desc: ConvolutionDescriptor,
    filter_desc: FilterDescriptor,
    bias_desc: TensorDescriptor,
    act_desc: ActivationDescriptor,

    algo: cudnnConvolutionFwdAlgo_t,

    // scratch memory
    work_mem: DeviceMem,

    highway_mem: DeviceMem,
    inter_mem: DeviceMem,
}

impl NetEvaluator {
    pub fn new(mut handle: CudnnHandle, def: NetDefinition, batch_size: i32) -> Self {
        let image_desc = TensorDescriptor::new(
            batch_size, def.tower_channels, 7, 7,
            cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        );
        let bias_desc = TensorDescriptor::new(
            1, def.tower_channels, 1, 1,
            cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        );
        let filter_desc = FilterDescriptor::new(
            def.tower_channels, def.tower_channels, 3, 3,
            cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
        );
        let conv_desc = ConvolutionDescriptor::new(
            1, 1, 1, 1, 1, 1, cudnnDataType_t::CUDNN_DATA_FLOAT,
        );
        let act_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_RELU, 0.0);

        let layers = (0..def.tower_depth).map(|_| {
            (
                ConvWeights::random(&handle, &filter_desc, &bias_desc),
                ConvWeights::random(&handle, &filter_desc, &bias_desc),
            )
        }).collect();

        let algo_info = find_conv_algorithms(
            &mut handle,
            &conv_desc, &filter_desc, &image_desc, &image_desc,
        )[0];

        let workspace = DeviceMem::alloc(algo_info.memory, handle.device());
        let highway_mem = DeviceMem::alloc(image_desc.size(), handle.device());
        let inter_mem = DeviceMem::alloc(image_desc.size(), handle.device());

        NetEvaluator {
            handle,
            batch_size,
            def,
            layers,
            image_desc,
            conv_desc,
            filter_desc,
            bias_desc,
            act_desc,
            algo: algo_info.algo,

            work_mem: workspace,
            highway_mem,
            inter_mem,
        }
    }

    /// Runs `data = net(data)`.
    pub fn eval(&mut self, data: &mut Vec<f32>) {
        assert_eq!(self.batch_size * self.def.tower_channels * 7 * 7, data.len() as i32);

        self.highway_mem.copy_from_host(cast_slice(data));

        for layer in &self.layers {
            let (first, second) = layer;

            run_conv_bias_res_activation(
                &mut self.handle,
                &self.act_desc,
                &self.conv_desc,
                self.algo,
                &mut self.work_mem,
                &self.filter_desc,
                &first.filter_mem,
                &self.image_desc,
                &self.highway_mem,
                ResType::Zero,
                &self.bias_desc,
                &first.bias_mem,
                &self.image_desc,
                &mut self.inter_mem,
            );

            run_conv_bias_res_activation(
                &mut self.handle,
                &self.act_desc,
                &self.conv_desc,
                self.algo,
                &mut self.work_mem,
                &self.filter_desc,
                &second.filter_mem,
                &self.image_desc,
                &self.inter_mem,
                ResType::Output,
                &self.bias_desc,
                &second.bias_mem,
                &self.image_desc,
                &mut self.highway_mem,
            );
        }

        // copy output back
        self.highway_mem.copy_to_host(cast_slice_mut(data));
    }
}