use std::ptr::null_mut;

use kn_cuda_sys::wrapper::handle::{CudaStream, Device};
use kn_cuda_sys::wrapper::rtc::args::KernelArgs;
use kn_cuda_sys::wrapper::rtc::core::{CuFunction, Dim3};
use kn_cuda_sys::wrapper::status::Status;
use kn_graph::dtype::DisplayCFloat;

use crate::autokernel::common::{
    c_array_string, c_nested_array_string, ceil_div, fill_replacements, KernelKey,
};
use crate::device_tensor::DeviceTensor;
use crate::shape::StridedShape;

#[derive(Debug)]
pub struct LayernormKernel<K> {
    input_shape: StridedShape,
    output_shape: StridedShape,

    _norm_axis: usize,
    static_size: usize,

    _eps: f32,
    _alpha0: f32,
    _alpha1: f32,
    _beta: f32,

    kernel: K,
}

const LAYERNORM_SOURCE: &str = include_str!("layernorm.cu");

impl LayernormKernel<KernelKey> {
    pub fn new(
        device: Device,
        input_shape: &StridedShape,
        output_shape: &StridedShape,
        norm_axis: usize,
        eps: f32,
        alpha_0: f32,
        alpha_1: f32,
        beta: f32,
    ) -> Self {
        assert_eq!(input_shape.shape(), output_shape.shape());

        let norm_size = input_shape.shape()[norm_axis];
        let static_size = input_shape.size() / norm_size;

        let input_static = input_shape.remove(norm_axis);
        let output_static = output_shape.remove(norm_axis);

        let static_dense = StridedShape::new_simple(input_static.shape().to_vec());

        let mut static_strides = [input_static.strides().to_vec(), output_static.strides().to_vec()];
        let mut static_dense_strides = static_dense.strides().to_vec();

        let norm_strides = [input_shape.strides()[norm_axis], output_shape.strides()[norm_axis]];

        // pad arrays to ensure they never become zero-sized
        static_strides[0].push(0);
        static_strides[1].push(0);
        static_dense_strides.push(1);

        let replacements = vec![
            ("$RANK$", format!("{}", input_shape.rank())),
            ("$STATIC_SIZE$", format!("{}", static_size)),
            ("$NORM_SIZE$", format!("{}", norm_size)),
            ("$EPS$", format!("{}", DisplayCFloat(eps as f64))),
            ("$ALPHA_0$", format!("{}", DisplayCFloat(alpha_0 as f64))),
            ("$ALPHA_1$", format!("{}", DisplayCFloat(alpha_1 as f64))),
            ("$BETA$", format!("{}", DisplayCFloat(beta as f64))),
            ("$STATIC_DENSE_STRIDES$", c_array_string(&static_dense_strides)),
            ("$STATIC_STRIDES$", c_nested_array_string(&static_strides)),
            ("$NORM_STRIDES$", c_array_string(&norm_strides)),
        ];

        // compile the kernel
        let source = fill_replacements(LAYERNORM_SOURCE, &replacements);
        let kernel = KernelKey {
            device,
            source,
            func_name: "layernorm_kernel".to_owned(),
        };

        // wrap everything up
        LayernormKernel {
            kernel,
            input_shape: input_shape.clone(),
            output_shape: output_shape.clone(),
            _norm_axis: norm_axis,
            static_size,
            _eps: eps,
            _alpha0: alpha_0,
            _alpha1: alpha_1,
            _beta: beta,
        }
    }
}

impl<T> LayernormKernel<T> {
    pub fn map_kernel<K>(&self, mut f: impl FnMut(&T) -> K) -> LayernormKernel<K> {
        LayernormKernel {
            input_shape: self.input_shape.clone(),
            output_shape: self.output_shape.clone(),
            _norm_axis: self._norm_axis,
            static_size: self.static_size,
            _eps: self._eps,
            _alpha0: self._alpha0,
            _alpha1: self._alpha1,
            _beta: self._beta,
            kernel: f(&self.kernel),
        }
    }
}

impl LayernormKernel<CuFunction> {
    pub unsafe fn run(
        &self,
        stream: &CudaStream,
        input0: &DeviceTensor,
        input1: Option<&DeviceTensor>,
        output: &DeviceTensor,
    ) {
        assert_eq!(input0.strided_shape(), &self.input_shape);
        if let Some(input1) = input1 {
            assert_eq!(input1.strided_shape(), &self.input_shape);
        }
        assert_eq!(output.strided_shape(), &self.output_shape);

        if self._alpha1 != 0.0 {
            assert_eq!(input1.is_some(), true);
        }

        let mut args = KernelArgs::new();
        args.push(input0.ptr().ptr());
        args.push(input1.map_or(null_mut(), |x| x.ptr().ptr()));
        args.push(output.ptr().ptr());
        let args = args.finish();

        //TODO see if these settings make sense for the typically larger layernorm sizes

        let warps = self.static_size;
        let warps_per_block = 4;
        let threads_per_warp = 32;

        let threads_per_block = (threads_per_warp * warps_per_block) as u32;
        let blocks = ceil_div((warps * threads_per_warp) as u32, threads_per_block as u32);

        self.kernel
            .launch_kernel(Dim3::single(blocks), Dim3::single(threads_per_block), 0, &stream, &args)
            .unwrap();
    }
}
