use std::io;

use bytemuck::cast_slice;
use itertools::Itertools;
use npyz::npz::NpzArchive;
use npyz::Order;

use cuda_sys::bindings::{cudnnDataType_t, cudnnTensorFormat_t};
use cuda_sys::wrapper::group::{Filter, Tensor};

use crate::graph::{Graph, Operation};
use cuda_sys::wrapper::handle::Device;
use std::fmt::{Debug, Formatter};

pub struct GraphParams {
    pub device: Device,
    pub filters: Vec<Filter>,
    pub biases: Vec<Tensor>,
}

impl GraphParams {
    /// Initialize matching parameters for Graph, with values all uninitialized.
    pub fn dummy(device: Device, graph: &Graph) -> Self {
        let mut filters = vec![];
        let mut biases = vec![];

        for value in graph.values() {
            let value_info = graph[value];

            match value_info.operation {
                Operation::Input => {}
                Operation::Conv { input, output_channels, kernel_size, .. } => {
                    let input_channels = graph[input].shape[1];
                    let filter = Filter::new(
                        output_channels, input_channels, kernel_size, kernel_size,
                        cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                        device
                    );
                    filters.push(filter)
                }
                Operation::Bias { channels, .. } => {
                    let bias = Tensor::new(
                        1, channels, 1, 1,
                        cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                        device
                    );
                    biases.push(bias);
                }
                Operation::Add { .. } => {}
                Operation::Relu { .. } => {}
            }
        }

        GraphParams { device, filters, biases }
    }

    fn empty(device: Device) -> Self {
        GraphParams {
            device,
            filters: Default::default(),
            biases: Default::default(),
        }
    }
}

pub fn load_params_from_npz<R: io::Read + io::Seek>(graph: &Graph, npz: &mut NpzArchive<R>, device: Device) -> GraphParams {
    let mut loader = Loader::new(npz, device);
    let mut params = GraphParams::empty(device);

    for value in graph.values() {
        match graph[value].operation {
            Operation::Input | Operation::Add { .. } | Operation::Relu { .. } => {}
            Operation::Conv { input, output_channels, kernel_size, padding: _, flat_weights } => {
                let [_, input_channels, w, h] = graph[input].shape;
                let filter_shape = [output_channels, input_channels, kernel_size, kernel_size];
                let flat_shape;

                let data_shape: &[i32] = if flat_weights {
                    flat_shape = [output_channels, input_channels * w * h];
                    &flat_shape
                } else {
                    &filter_shape
                };

                let filter = loader.next_filter(filter_shape, data_shape);
                params.filters.push(filter);
            }
            Operation::Bias { input: _, channels } => {
                let bias = loader.next_bias(channels);
                params.biases.push(bias);
            }
        }
    }

    assert!(loader.is_done(), "{} leftover parameters", loader.max - loader.next);
    params
}

struct Loader<'a, R: io::Read + io::Seek> {
    device: Device,
    npz: &'a mut NpzArchive<R>,

    max: usize,
    next: usize,
}

impl<'a, R: io::Read + io::Seek> Loader<'a, R> {
    fn new(npz: &'a mut NpzArchive<R>, device: Device) -> Self {
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
        assert_eq!(&shape, result.shape(), "param {}: shape mismatch, maybe the graph doesn't math the parameter file?", next);

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

    fn is_done(&self) -> bool {
        self.next == self.max
    }
}

impl Debug for GraphParams {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let GraphParams { device, filters, biases } = self;

        writeln!(f, "GraphParams {{")?;
        writeln!(f, "  device: {:?},", device)?;

        writeln!(f, "  filters: [")?;
        for (i, filter) in filters.iter().enumerate() {
            writeln!(f, "    {} -> {:?},", i, filter)?;
        }
        writeln!(f, "  ],")?;

        writeln!(f, "  biases: [")?;
        for (i, bias) in biases.iter().enumerate() {
            writeln!(f, "    {} -> {:?},", i, bias)?;
        }
        writeln!(f, "  ],")?;

        Ok(())
    }
}