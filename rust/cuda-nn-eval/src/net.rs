use bytemuck::{cast_slice, cast_slice_mut};

use cuda_sys::bindings::{cudnnActivationMode_t, cudnnDataType_t, cudnnTensorFormat_t};
use cuda_sys::wrapper::descriptor::{ActivationDescriptor, ConvolutionDescriptor};
use cuda_sys::wrapper::group::{Convolution, Filter, Tensor};
use cuda_sys::wrapper::handle::CudnnHandle;
use cuda_sys::wrapper::operation::{ResType, run_conv_bias_res_activation};

#[derive(Copy, Clone)]
pub struct ResNetShape {
    pub board_size: i32,
    pub input_channels: i32,

    pub tower_depth: usize,
    pub tower_channels: i32,

    pub wdl_hidden_size: i32,
    pub policy_channels: i32,
}

pub struct ResNetParams<F, B> {
    initial_conv: WeightBias<F, B>,

    tower: Vec<BlockParams<F, B>>,

    policy_conv: WeightBias<F, B>,
    wdl_initial_conv: WeightBias<F, B>,
    wdl_hidden_conv: WeightBias<F, B>,
    wdl_output_conv: WeightBias<F, B>,
}

impl ResNetParams<Filter, Tensor> {
    pub fn dummy(shape: ResNetShape, device: i32) -> Self {
        let c = shape.tower_channels;

        ResNetParams {
            initial_conv: WeightBias::dummy(c, shape.input_channels, 3, device),
            tower: (0..shape.tower_depth).map(|_| BlockParams::dummy(shape, device)).collect(),
            policy_conv: WeightBias::dummy(shape.policy_channels, c, 3, device),
            wdl_initial_conv: WeightBias::dummy(1, c, 1, device),
            wdl_hidden_conv: WeightBias::dummy(shape.wdl_hidden_size, 1, shape.board_size, device),
            wdl_output_conv: WeightBias::dummy(3, shape.wdl_hidden_size, 1, device),
        }
    }
}

pub struct BlockParams<F, B> {
    first: WeightBias<F, B>,
    second: WeightBias<F, B>,
}

impl BlockParams<Filter, Tensor> {
    fn dummy(shape: ResNetShape, device: i32) -> Self {
        BlockParams {
            first: WeightBias::dummy(shape.tower_channels, shape.tower_channels, 3, device),
            second: WeightBias::dummy(shape.tower_channels, shape.tower_channels, 3, device),
        }
    }
}

pub struct WeightBias<F, B> {
    filter: F,
    bias: B,
}

impl WeightBias<Filter, Tensor> {
    fn dummy(k: i32, c: i32, k_size: i32, device: i32) -> Self {
        WeightBias {
            filter: Filter::new(k, c, k_size, k_size, cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, device),
            bias: Tensor::new(1, k, 1, 1, cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, device),
        }
    }
}

pub struct NetEvaluator {
    handle: CudnnHandle,
    batch_size: i32,

    shape: ResNetShape,
    params: ResNetParams<Filter, Tensor>,

    activation_relu: ActivationDescriptor,
    activation_none: ActivationDescriptor,

    initial_conv: Convolution,
    tower_conv: Convolution,
    policy_conv: Convolution,
    wdl_initial_conv: Convolution,
    wdl_hidden_conv: Convolution,
    wdl_output_conv: Convolution,

    input: Tensor,
    highway: Tensor,
    inter: Tensor,

    policy_output: Tensor,
    wdl_initial: Tensor,
    wdl_hidden: Tensor,
    wdl_output: Tensor,
}

impl NetEvaluator {
    pub fn new(mut handle: CudnnHandle, shape: ResNetShape, params: ResNetParams<Filter, Tensor>, batch_size: i32) -> Self {
        // abbreviations
        let n = batch_size;
        let s = shape.board_size;
        let c = shape.tower_channels;

        let device = handle.device();
        let data_type = cudnnDataType_t::CUDNN_DATA_FLOAT;
        let format = cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;

        // activation

        // tensors
        let input = Tensor::new(n, shape.input_channels, s, s, data_type, format, device);
        let highway = Tensor::new(n, c, s, s, data_type, format, device);
        let inter = Tensor::new(n, c, s, s, data_type, format, device);
        let policy_output = Tensor::new(n, shape.policy_channels, s, s, data_type, format, device);
        let wdl_initial = Tensor::new(n, 1, s, s, data_type, format, device);
        let wdl_hidden = Tensor::new(n, shape.wdl_hidden_size, 1, 1, data_type, format, device);
        let wdl_output = Tensor::new(n, 3, 1, 1, data_type, format, device);

        // convolutions
        let initial_conv = Convolution::with_best_algo(
            &mut handle,
            ConvolutionDescriptor::new(1, 1, 1, 1, 1, 1, data_type),
            &params.initial_conv.filter.desc,
            &input.desc,
            &highway.desc,
        );

        // TODO expand this to support changing channel sizes
        let tower_conv = Convolution::with_best_algo(
            &mut handle,
            ConvolutionDescriptor::new(1, 1, 1, 1, 1, 1, data_type),
            &params.tower[0].first.filter.desc,
            &highway.desc,
            &inter.desc,
        );

        let policy_conv = Convolution::with_best_algo(
            &mut handle,
            ConvolutionDescriptor::new(1, 1, 1, 1, 1, 1, data_type),
            &params.policy_conv.filter.desc,
            &highway.desc,
            &policy_output.desc,
        );

        let wdl_initial_conv = Convolution::with_best_algo(
            &mut handle,
            ConvolutionDescriptor::new(0, 0, 1, 1, 1, 1, data_type),
            &params.wdl_initial_conv.filter.desc,
            &highway.desc,
            &wdl_initial.desc,
        );

        let wdl_hidden_conv = Convolution::with_best_algo(
            &mut handle,
            ConvolutionDescriptor::new(0, 0, 1, 1, 1, 1, data_type),
            &params.wdl_hidden_conv.filter.desc,
            &wdl_initial.desc,
            &wdl_hidden.desc,
        );

        let wdl_final_conv = Convolution::with_best_algo(
            &mut handle,
            ConvolutionDescriptor::new(0, 0, 1, 1, 1, 1, data_type),
            &params.wdl_output_conv.filter.desc,
            &wdl_hidden.desc,
            &wdl_output.desc,
        );

        NetEvaluator {
            batch_size,

            handle,
            shape,
            params,

            activation_relu: ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_RELU, 0.0),
            activation_none: ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY, 0.0),

            initial_conv,
            tower_conv,
            policy_conv,
            wdl_initial_conv,
            wdl_hidden_conv,
            wdl_output_conv: wdl_final_conv,

            input,
            highway,
            inter,
            policy_output,
            wdl_initial,
            wdl_hidden,
            wdl_output,
        }
    }

    /// Runs `data = net(data)`.
    pub fn eval(&mut self, input: &[f32], output_wdl: &mut [f32], output_policy: &mut [f32]) {
        let batch_size = self.batch_size;
        let shape = self.shape;
        let w = shape.board_size;

        assert_eq!(input.len() as i32, batch_size * shape.input_channels * w * w, "input_size mismatch");
        assert_eq!(output_wdl.len() as i32, batch_size * 3, "output_wdl size mismatch");
        assert_eq!(output_policy.len() as i32, batch_size * shape.policy_channels * w * w, "policy size mismatch");

        let handle = &mut self.handle;

        // copy input to device
        self.input.mem.copy_from_host(cast_slice(input));

        // initial
        run_conv_bias_res_activation(
            handle,
            &self.activation_relu,
            &self.initial_conv.desc,
            self.initial_conv.algo,
            &mut self.initial_conv.workspace,
            &self.params.initial_conv.filter.desc,
            &self.params.initial_conv.filter.mem,
            &self.input.desc,
            &self.input.mem,
            ResType::Zero,
            &self.params.initial_conv.bias.desc,
            &self.params.initial_conv.bias.mem,
            &self.highway.desc,
            &mut self.highway.mem,
        );

        // tower
        for layer in &self.params.tower {
            let BlockParams { first, second } = layer;

            run_conv_bias_res_activation(
                handle,
                &self.activation_relu,
                &self.tower_conv.desc,
                self.tower_conv.algo,
                &mut self.tower_conv.workspace,
                &first.filter.desc,
                &first.filter.mem,
                &self.highway.desc,
                &self.highway.mem,
                ResType::Zero,
                &first.bias.desc,
                &first.bias.mem,
                &self.inter.desc,
                &mut self.inter.mem,
            );

            run_conv_bias_res_activation(
                handle,
                &self.activation_relu,
                &self.tower_conv.desc,
                self.tower_conv.algo,
                &mut self.tower_conv.workspace,
                &second.filter.desc,
                &second.filter.mem,
                &self.inter.desc,
                &self.inter.mem,
                ResType::Output,
                &second.bias.desc,
                &second.bias.mem,
                &self.highway.desc,
                &mut self.highway.mem,
            );
        }

        // policy
        run_conv_bias_res_activation(
            handle,
            &self.activation_none,
            &self.policy_conv.desc,
            self.policy_conv.algo,
            &mut self.policy_conv.workspace,
            &self.params.policy_conv.filter.desc,
            &self.params.policy_conv.filter.mem,
            &self.highway.desc,
            &self.highway.mem,
            ResType::Zero,
            &self.params.policy_conv.bias.desc,
            &self.params.policy_conv.bias.mem,
            &self.policy_output.desc,
            &mut self.policy_output.mem,
        );

        // wdl
        run_conv_bias_res_activation(
            handle,
            &self.activation_relu,
            &self.wdl_initial_conv.desc,
            self.wdl_initial_conv.algo,
            &mut self.wdl_initial_conv.workspace,
            &self.params.wdl_initial_conv.filter.desc,
            &self.params.wdl_initial_conv.filter.mem,
            &self.highway.desc,
            &self.highway.mem,
            ResType::Zero,
            &self.params.wdl_initial_conv.bias.desc,
            &self.params.wdl_initial_conv.bias.mem,
            &self.wdl_initial.desc,
            &mut self.wdl_initial.mem,
        );

        run_conv_bias_res_activation(
            handle,
            &self.activation_relu,
            &self.wdl_hidden_conv.desc,
            self.wdl_hidden_conv.algo,
            &mut self.wdl_hidden_conv.workspace,
            &self.params.wdl_hidden_conv.filter.desc,
            &self.params.wdl_hidden_conv.filter.mem,
            &self.wdl_initial.desc,
            &self.wdl_initial.mem,
            ResType::Zero,
            &self.params.wdl_hidden_conv.bias.desc,
            &self.params.wdl_hidden_conv.bias.mem,
            &self.wdl_hidden.desc,
            &mut self.wdl_hidden.mem,
        );

        run_conv_bias_res_activation(
            handle,
            &self.activation_none,
            &self.wdl_output_conv.desc,
            self.wdl_output_conv.algo,
            &mut self.wdl_output_conv.workspace,
            &self.params.wdl_output_conv.filter.desc,
            &self.params.wdl_output_conv.filter.mem,
            &self.wdl_hidden.desc,
            &self.wdl_hidden.mem,
            ResType::Zero,
            &self.params.wdl_output_conv.bias.desc,
            &self.params.wdl_output_conv.bias.mem,
            &self.wdl_output.desc,
            &mut self.wdl_output.mem,
        );

        // copy output back from device
        self.wdl_output.mem.copy_to_host(cast_slice_mut(output_wdl));
        self.policy_output.mem.copy_to_host(cast_slice_mut(output_policy));
    }
}
