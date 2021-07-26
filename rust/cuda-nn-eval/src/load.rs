use std::io;

use bytemuck::cast_slice;
use itertools::Itertools;
use npyz::Order;
use npyz::npz::NpzArchive;

use cuda_sys::bindings::{cudnnDataType_t, cudnnTensorFormat_t};
use cuda_sys::wrapper::group::{Filter, Tensor};

use crate::net::{BlockParams, ConvParams, ResNetParams, ResNetShape};

pub fn load_net_params<R>(shape: ResNetShape, npz: &mut NpzArchive<R>, device: i32) -> ResNetParams<Filter, Tensor> where R: io::Read + io::Seek {
    let mut loader = Loader::new(npz, device);

    let c = shape.tower_channels;
    let s = shape.board_size;

    let params = ResNetParams {
        initial_conv: loader.next_conv([c, shape.input_channels, 3, 3]),
        tower: (0..shape.tower_depth).map(|_| {
            BlockParams {
                first: loader.next_conv([c, c, 3, 3]),
                second: loader.next_conv([c, c, 3, 3]),
            }
        }).collect(),

        wdl_initial_conv: loader.next_conv([1, c, 1, 1]),
        wdl_hidden_conv: loader.next_general_conv(
            [shape.wdl_hidden_size, 1, s, s],
            &[shape.wdl_hidden_size, s * s]
        ),
        wdl_output_conv: loader.next_general_conv(
            [3, shape.wdl_hidden_size, 1, 1],
            &[3, shape.wdl_hidden_size]
        ),

        policy_conv: loader.next_conv([shape.policy_channels, c, 1, 1]),
    };

    assert!(loader.is_done(), "{} leftover parameters", loader.max - loader.next);

    params
}

struct Loader<'a, R: io::Read + io::Seek> {
    device: i32,
    npz: &'a mut NpzArchive<R>,

    max: usize,
    next: usize,
}

impl<'a, R: io::Read + io::Seek> Loader<'a, R> {
    fn new(npz: &'a mut NpzArchive<R>, device: i32) -> Self {
        let max = npz.array_names().count();
        Loader { device, next: 0, max, npz }
    }

    fn next(&mut self, shape: &[i32]) -> Vec<f32> {
        let next = self.next;
        assert!(next < self.max, "Ran out of parameters at index {}", self.next);
        self.next += 1;

        let result = self.npz.by_name(&format!("arr_{}", next))
            .unwrap_or_else(|_| panic!("Failed while trying to read arr_{}", next))
            .unwrap_or_else(|| panic!("Missing arr_{}", next));

        assert_eq!(Order::C, result.order(), "param {}: must be in C-order", next);
        let shape = shape.iter().map(|&x| x as u64).collect_vec();
        assert_eq!(&shape, result.shape(), "param {}: shape mismatch", next);

        result.into_vec()
            .unwrap_or_else(|_| panic!("param {}: failed to convert to vec, maybe type mismatch?", next))
    }

    fn next_filter(&mut self, filter_shape: [i32; 4], data_shape: &[i32]) -> Filter {
        let data = self.next(data_shape);

        let mut filter = Filter::new(
            filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3],
            cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, self.device,
        );
        filter.mem.copy_from_host(cast_slice(&data));
        filter
    }

    fn next_bias(&mut self, c: i32) -> Tensor {
        let data = self.next(&[c]);

        let mut tensor = Tensor::new(
            1, c, 1, 1,
            cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, self.device,
        );
        tensor.mem.copy_from_host(cast_slice(&data));
        tensor
    }

    /// Load the params for any layer that can be represented as a convolution,
    /// flattening followed by a fully connected layer.
    fn next_general_conv(&mut self, filter_shape: [i32; 4], data_shape: &[i32]) -> ConvParams<Filter, Tensor> {
        ConvParams {
            filter: self.next_filter(filter_shape, data_shape),
            bias: self.next_bias(filter_shape[0]),
        }
    }

    /// Load the params for a real convolution.
    fn next_conv(&mut self, shape: [i32; 4]) -> ConvParams<Filter, Tensor> {
        self.next_general_conv(shape, &shape)
    }

    fn is_done(&self) -> bool {
        self.next == self.max
    }
}
