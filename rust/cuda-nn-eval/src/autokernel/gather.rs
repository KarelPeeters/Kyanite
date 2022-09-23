use cuda_sys::wrapper::handle::{CudaStream, Device};
use cuda_sys::wrapper::rtc::args::KernelArgs;
use cuda_sys::wrapper::rtc::core::{CuFunction, Dim3};

use crate::autokernel::common::{
    c_array_string, c_nested_array_string, ceil_div, compile_cached_kernel, fill_replacements, KernelKey,
};
use crate::device_tensor::DeviceTensor;
use crate::shape::StridedShape;

#[derive(Debug)]
pub struct GatherKernel {
    input_shape: StridedShape,
    indices_shape: StridedShape,
    output_shape: StridedShape,

    _axis: usize,

    function: CuFunction,
}

const GATHER_SOURCE: &str = include_str!("gather.cu");

impl GatherKernel {
    pub fn new(
        device: Device,
        input_shape: &StridedShape,
        indices_shape: &StridedShape,
        output_shape: &StridedShape,
        axis: usize,
    ) -> Self {
        assert_eq!(indices_shape.rank(), 1);
        let indices_size = indices_shape.size();
        let indices_stride = indices_shape.strides()[0];
        assert_eq!(input_shape.rank(), output_shape.rank());
        assert_eq!(indices_size, output_shape.shape()[axis]);

        let kept_input_shape = input_shape.remove(axis);
        let kept_output_shape = output_shape.remove(axis);
        assert_eq!(kept_input_shape.shape(), kept_output_shape.shape());

        let mut kept_strides = vec![
            kept_input_shape.strides().to_vec(),
            kept_output_shape.strides().to_vec(),
        ];

        let kept_shape_dense = StridedShape::new_simple(kept_input_shape.shape().to_vec());
        let mut kept_strides_dense = kept_shape_dense.strides().to_vec();

        // pad arrays to ensure they never become zero-sized
        kept_strides[0].push(0);
        kept_strides[1].push(0);
        kept_strides_dense.push(1);

        // to avoid a division by zero warning in the compiled kernel
        let indices_size_div = if indices_size == 0 { 1 } else { indices_size };

        let replacements = vec![
            ("$RANK$", format!("{}", input_shape.rank())),
            ("$KEPT_STRIDES_DENSE$", c_array_string(&kept_strides_dense)),
            ("$KEPT_STRIDES$", c_nested_array_string(&kept_strides)),
            ("$INPUT_AXIS_SIZE$", format!("{}", input_shape.shape()[axis])),
            ("$INPUT_AXIS_STRIDE$", format!("{}", input_shape.strides()[axis])),
            ("$OUTPUT_AXIS_STRIDE$", format!("{}", output_shape.strides()[axis])),
            ("$INDICES_SIZE_DIV$", format!("{}", indices_size_div)),
            ("$INDICES_STRIDE$", format!("{}", indices_stride)),
            ("$OUTPUT_SIZE$", format!("{}", output_shape.size())),
        ];

        let source = fill_replacements(GATHER_SOURCE, &replacements);

        let key = KernelKey {
            device,
            source,
            func_name: "gather_kernel".to_owned(),
        };
        let function = compile_cached_kernel(key);

        // wrap everything up
        GatherKernel {
            function,
            input_shape: input_shape.clone(),
            indices_shape: indices_shape.clone(),
            output_shape: output_shape.clone(),
            _axis: axis,
        }
    }

    pub unsafe fn run(&self, stream: &CudaStream, input: &DeviceTensor, indices: &DeviceTensor, output: &DeviceTensor) {
        assert_eq!(input.strided_shape(), &self.input_shape);
        assert_eq!(indices.strided_shape(), &self.indices_shape);
        assert_eq!(output.strided_shape(), &self.output_shape);

        let mut args = KernelArgs::new();
        args.push(input.ptr().ptr());
        args.push(indices.ptr().ptr());
        args.push(output.ptr().ptr());
        let args = args.finish();

        // TODO pick good values for these
        let items_per_thread = 64;
        let threads_per_block = 64;

        let items = self.output_shape.size();
        let blocks = ceil_div(items as u32, items_per_thread * threads_per_block);

        self.function
            .launch_kernel(Dim3::single(blocks), Dim3::single(threads_per_block), 0, &stream, &args);
    }
}
