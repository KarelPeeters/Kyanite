use kn_cuda_sys::wrapper::handle::{CudaDevice, CudaStream};
use kn_cuda_sys::wrapper::rtc::args::KernelArgs;
use kn_cuda_sys::wrapper::rtc::core::{CuFunction, Dim3};
use kn_cuda_sys::wrapper::status::Status;
use kn_graph::dtype::DisplayCFloat;

use crate::autokernel::common::{
    c_array_string, c_nested_array_string, ceil_div, compile_cached_kernel, fill_replacements, KernelKey,
};
use crate::device_tensor::DeviceTensor;
use crate::shape::StridedShape;

#[derive(Debug)]
pub struct SoftmaxKernel {
    input_shape: StridedShape,
    output_shape: StridedShape,

    _softmax_axis: usize,
    _input_scale: f32,
    static_size: usize,

    function: CuFunction,
}

const SOFTMAX_SOURCE: &str = include_str!("softmax.cu");

impl SoftmaxKernel {
    pub fn new(
        device: CudaDevice,
        input_shape: &StridedShape,
        output_shape: &StridedShape,
        softmax_axis: usize,
        input_scale: f32,
    ) -> Self {
        assert_eq!(input_shape.shape(), output_shape.shape());

        let softmax_size = input_shape.shape()[softmax_axis];
        let static_size = input_shape.size() / softmax_size;

        let input_static = input_shape.remove(softmax_axis);
        let output_static = output_shape.remove(softmax_axis);

        let static_dense = StridedShape::new_simple(input_static.shape().to_vec());

        let mut static_strides = [input_static.strides().to_vec(), output_static.strides().to_vec()];
        let mut static_dense_strides = static_dense.strides().to_vec();

        let softmax_strides = [
            input_shape.strides()[softmax_axis],
            output_shape.strides()[softmax_axis],
        ];

        // pad arrays to ensure they never become zero-sized
        static_strides[0].push(0);
        static_strides[1].push(0);
        static_dense_strides.push(1);

        let replacements = vec![
            ("$RANK$", format!("{}", input_shape.rank())),
            ("$STATIC_SIZE$", format!("{}", static_size)),
            ("$SOFTMAX_SIZE$", format!("{}", softmax_size)),
            ("$INPUT_SCALE$", format!("{}", DisplayCFloat(input_scale as f64))),
            ("$STATIC_DENSE_STRIDES$", c_array_string(&static_dense_strides)),
            ("$STATIC_STRIDES$", c_nested_array_string(&static_strides)),
            ("$SOFTMAX_STRIDES$", c_array_string(&softmax_strides)),
        ];

        // compile the kernel
        let source = fill_replacements(SOFTMAX_SOURCE, &replacements);
        let key = KernelKey {
            device,
            source,
            func_name: "softmax_kernel".to_owned(),
        };
        let function = compile_cached_kernel(key);

        // wrap everything up
        SoftmaxKernel {
            function,
            input_shape: input_shape.clone(),
            output_shape: output_shape.clone(),
            _softmax_axis: softmax_axis,
            _input_scale: input_scale,
            static_size,
        }
    }

    pub unsafe fn run(&self, stream: &CudaStream, input: &DeviceTensor, output: &DeviceTensor) {
        assert_eq!(input.strided_shape(), &self.input_shape);
        assert_eq!(output.strided_shape(), &self.output_shape);

        let mut args = KernelArgs::new();
        args.push(input.ptr().ptr());
        args.push(output.ptr().ptr());
        let args = args.finish();

        let warps = self.static_size;
        let warps_per_block = 4;
        let threads_per_warp = 32;

        let threads_per_block = (threads_per_warp * warps_per_block) as u32;
        let blocks = ceil_div((warps * threads_per_warp) as u32, threads_per_block as u32);

        self.function
            .launch_kernel(Dim3::single(blocks), Dim3::single(threads_per_block), 0, &stream, &args)
            .unwrap();
    }
}
