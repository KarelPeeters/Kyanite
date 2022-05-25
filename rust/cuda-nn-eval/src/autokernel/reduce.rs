use itertools::Itertools;

use cuda_sys::wrapper::handle::{ComputeCapability, CudaStream};
use cuda_sys::wrapper::rtc::args::KernelArgs;
use cuda_sys::wrapper::rtc::core::{CuFunction, Dim3};

use crate::autokernel::common::{
    c_array_string, c_nested_array_string, ceil_div, compile_cached_kernel, fill_replacements, KernelKey,
};
use crate::device_tensor::DeviceTensor;
use crate::shape::StridedShape;

#[derive(Debug)]
pub struct ReduceKernel {
    _code: ReduceCode,
    _reduced_axes: Vec<usize>,

    capability: ComputeCapability,
    function: CuFunction,

    input_shape: StridedShape,
    output_shape: StridedShape,
}

const REDUCE_SOURCE: &str = include_str!("reduce.cu");

#[derive(Debug, Clone)]
pub struct ReduceCode {
    pub ty: String,
    pub identity: String,
    pub operation: String,
    pub post_process: String,
}

impl ReduceKernel {
    pub fn new(
        capability: ComputeCapability,
        code: ReduceCode,
        input_shape: &StridedShape,
        output_shape: &StridedShape,
        reduced_axes: &[usize],
    ) -> Self {
        // check that axes are unique and in-bounds
        assert_eq!(
            reduced_axes.iter().unique().count(),
            reduced_axes.len(),
            "Reduced axes must be unique, got {:?}",
            reduced_axes
        );
        for &axis in reduced_axes {
            assert!(
                axis < input_shape.rank(),
                "Reduced axis out of bounds for shape {:?}",
                input_shape
            );
        }

        // split strides and shapes
        let mut input_kept_shape = vec![];
        let mut input_reduced_shape = vec![];
        let mut input_kept_strides = vec![];
        let mut input_reduced_strides = vec![];

        for axis in 0..input_shape.rank() {
            let size = input_shape.shape()[axis];
            let stride = input_shape.strides()[axis];

            if reduced_axes.contains(&axis) {
                input_reduced_shape.push(size);
                input_reduced_strides.push(stride);
            } else {
                input_kept_shape.push(size);
                input_kept_strides.push(stride);
            }
        }

        // check that things make sense
        let kept_size: usize = input_kept_shape.iter().copied().product();
        let reduction_size: usize = input_reduced_shape.iter().copied().product();

        assert_eq!(input_kept_shape, output_shape.shape(), "Output shape mismatch");
        assert_eq!(kept_size, output_shape.size());
        assert_eq!(input_shape.size(), kept_size * reduction_size);

        // build replacements
        let kept_shape_dense = StridedShape::new_simple(input_kept_shape.clone());
        let reduced_shape_dense = StridedShape::new_simple(input_reduced_shape.clone());

        // pad arrays to ensure they never become zero-sized
        let mut kept_stides_dense = kept_shape_dense.strides().to_vec();
        kept_stides_dense.push(0);
        input_kept_strides.push(0);
        let mut output_kept_strides = output_shape.strides().to_vec();
        output_kept_strides.push(0);

        let replacements = vec![
            ("$KEPT_RANK$", format!("{}", input_kept_shape.len())),
            ("$REDUCED_RANK$", format!("{}", input_reduced_shape.len())),
            ("$KEPT_SIZE$", format!("{}", kept_size)),
            ("$REDUCTION_SIZE$", format!("{}", reduction_size)),
            ("$KEPT_STRIDES_DENSE$", c_array_string(&kept_stides_dense)),
            ("$REDUCED_STRIDES_DENSE$", c_array_string(reduced_shape_dense.strides())),
            (
                "$KEPT_STRIDES$",
                c_nested_array_string(&[input_kept_strides, output_kept_strides]),
            ),
            ("$REDUCED_STRIDES$", c_array_string(&input_reduced_strides)),
            ("$TYPE$", code.ty.clone()),
            ("$IDENTITY$", code.identity.clone()),
            ("$OPERATION$", code.operation.clone()),
            ("$POST_PROCESS$", code.post_process.clone()),
        ];

        // compile the kernel
        let source = fill_replacements(REDUCE_SOURCE, &replacements);
        let key = KernelKey {
            capability,
            source,
            func_name: "reduce_kernel".to_owned(),
        };
        let function = compile_cached_kernel(key);

        // wrap everything up
        ReduceKernel {
            capability,
            function,
            _code: code,
            _reduced_axes: reduced_axes.to_owned(),
            input_shape: input_shape.clone(),
            output_shape: output_shape.clone(),
        }
    }

    pub unsafe fn run(&self, stream: &CudaStream, input: &DeviceTensor, output: &DeviceTensor) {
        assert_eq!(stream.device().compute_capability(), self.capability);

        assert_eq!(input.shape(), &self.input_shape);
        assert_eq!(output.shape(), &self.output_shape);

        let mut args = KernelArgs::new();
        args.push(input.ptr().ptr());
        args.push(output.ptr().ptr());
        let args = args.finish();

        let warps = self.output_shape.size();
        // TODO see what the effect of increasing this is
        let warps_per_block = 16;
        let threads_per_warp = 32;

        let threads_per_block = (threads_per_warp * warps_per_block) as u32;
        let blocks = ceil_div((warps * threads_per_warp) as u32, threads_per_block as u32);

        self.function
            .launch_kernel(Dim3::single(blocks), Dim3::single(threads_per_block), 0, &stream, &args);
    }
}
