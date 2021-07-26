use crate::bindings::{cudnnConvolutionFwdAlgo_t, cudnnDataType_t, cudnnTensorFormat_t};
use crate::wrapper::descriptor::{ConvolutionDescriptor, FilterDescriptor, TensorDescriptor};
use crate::wrapper::handle::CudnnHandle;
use crate::wrapper::mem::DeviceMem;
use crate::wrapper::operation::find_conv_algorithms;

#[derive(Debug)]
pub struct Tensor {
    pub desc: TensorDescriptor,
    pub mem: DeviceMem,
}

impl Tensor {
    pub fn new(n: i32, c: i32, h: i32, w: i32, data_type: cudnnDataType_t, format: cudnnTensorFormat_t, device: i32) -> Self {
        let desc = TensorDescriptor::new(n, c, h, w, data_type, format);
        let mem = DeviceMem::alloc(desc.size(), device);
        Tensor { desc, mem }
    }
}

#[derive(Debug)]
pub struct Filter {
    pub desc: FilterDescriptor,
    pub mem: DeviceMem,
}

impl Filter {
    pub fn new(k: i32, c: i32, h: i32, w: i32, data_type: cudnnDataType_t, format: cudnnTensorFormat_t, device: i32) -> Self {
        let desc = FilterDescriptor::new(k, c, h, w, data_type, format);
        let mem = DeviceMem::alloc(desc.size(), device);
        Filter { desc, mem }
    }
}

#[derive(Debug)]
pub struct Convolution {
    pub desc: ConvolutionDescriptor,
    pub algo: cudnnConvolutionFwdAlgo_t,
    pub workspace: DeviceMem,
}

impl Convolution {
    pub fn new_with_best_algo(
        handle: &mut CudnnHandle,
        conv: ConvolutionDescriptor,
        filter: &FilterDescriptor,
        input: &TensorDescriptor,
        output: &TensorDescriptor,
    ) -> Self {
        let output_shape = conv.output_shape(input, filter);
        if output_shape != output.shape() {
            panic!(
                "Shape mismatch: (input: {:?}, filter: {:?}) -> {:?} != {:?}",
                input.shape(), filter.shape(), output_shape, output.shape()
            )
        }

        let algos = find_conv_algorithms(handle, &conv, filter, input, output);
        let algo = algos
            .get(0)
            .expect("No algorithm found");

        let workspace = DeviceMem::alloc(algo.memory, handle.device());

        Convolution { desc: conv, algo: algo.algo, workspace }
    }
}