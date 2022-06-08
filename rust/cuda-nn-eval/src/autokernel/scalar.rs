use std::fmt::Write;

use itertools::zip_eq;

use cuda_sys::wrapper::handle::{CudaStream, Device};
use cuda_sys::wrapper::rtc::args::KernelArgs;
use cuda_sys::wrapper::rtc::core::{CuFunction, Dim3};

use crate::autokernel::common::{
    c_array_string, c_nested_array_string, ceil_div, compile_cached_kernel, fill_replacements, KernelKey,
};
use crate::device_tensor::DeviceTensor;
use crate::shape::StridedShape;

/// An instance of a scalar/elementwise kernel. Can be build for any operation, rank, strides, and sizes.
/// The first axis is runtime-dynamic without recompiling the kernel.
#[derive(Debug)]
pub struct ScalarKernel {
    #[allow(dead_code)]
    operation: String,

    inner_size: usize,
    inner_shape: Vec<usize>,
    operand_types: Vec<String>,
    operand_strides: Vec<Vec<isize>>,

    function: CuFunction,
}

const SCALAR_SOURCE: &str = include_str!("scalar.cu");

impl ScalarKernel {
    /// Compile an instance of a new scalar kernel.
    ///
    /// `operation` has the format `*x0 = *x1 + *x2;`.
    pub fn new(
        device: Device,
        operation: &str,
        inner_shape: Vec<usize>,
        operand_types: Vec<String>,
        operand_strides: Vec<Vec<isize>>,
    ) -> Self {
        assert!(operand_types.len() > 0);
        assert_eq!(operand_strides.len(), operand_types.len());
        for stride in &operand_strides {
            assert_eq!(stride.len(), inner_shape.len() + 1);
        }

        assert!(
            operation.trim_end().ends_with(";"),
            "Operation should end with ';', got {:?}",
            operation
        );

        let full_operation = build_operation(&operand_types, operation);

        let mut full_shape = vec![0];
        full_shape.extend_from_slice(&inner_shape);

        let dense = StridedShape::new_simple(full_shape.to_vec());

        let replacements = vec![
            ("$RANK$", format!("{}", dense.rank())),
            ("$OPERANDS$", format!("{}", operand_strides.len())),
            ("$STRIDES_DENSE$", c_array_string(dense.strides())),
            ("$STRIDES$", c_nested_array_string(&operand_strides)),
            ("$OPERATION$", full_operation.to_owned()),
        ];
        let source = fill_replacements(SCALAR_SOURCE, &replacements);

        let key = KernelKey {
            device,
            source,
            func_name: "scalar_kernel".to_owned(),
        };

        let function = compile_cached_kernel(key);
        let inner_size = inner_shape.iter().product();

        ScalarKernel {
            operation: operation.to_owned(),
            function,
            inner_size,
            inner_shape,
            operand_types,
            operand_strides,
        }
    }

    /// Wrapper around [Self::new] that's a bit easier to use if you know the full shape of the operands up front.
    pub fn new_for_shapes(device: Device, operation: &str, shapes: &[StridedShape]) -> Self {
        assert!(shapes.len() > 0);
        let expected_shape = shapes[0].shape();
        assert!(expected_shape.len() > 0);

        for shape in shapes {
            assert_eq!(shape.shape(), expected_shape);
        }

        let inner_shape = shapes[0].shape()[1..].to_vec();
        let operand_types = vec![String::from("float"); shapes.len()];
        let operand_strides = shapes.iter().map(|s| s.strides().to_vec()).collect();

        Self::new(device, operation, inner_shape, operand_types, operand_strides)
    }

    pub unsafe fn run(&self, stream: &CudaStream, tensors: &[DeviceTensor]) {
        let items_per_thread = 64;
        let threads_per_block = 64;
        self.run_custom(stream, tensors, items_per_thread, threads_per_block);
    }

    pub unsafe fn run_custom(
        &self,
        stream: &CudaStream,
        tensors: &[DeviceTensor],
        items_per_thread: u32,
        threads_per_block: u32,
    ) {
        assert_eq!(tensors.len(), self.operand_types.len());
        //TODO verify tensor types once that's implemented

        let batch_size = tensors[0].shape().shape()[0];

        let mut args = KernelArgs::new();
        args.push_int(batch_size as i32);

        for (expected_strides, tensor) in zip_eq(&self.operand_strides, tensors) {
            assert_eq!(1 + self.inner_shape.len(), tensor.shape().rank());
            assert_eq!(batch_size, tensor.shape().shape()[0]);
            assert_eq!(self.inner_shape, tensor.shape().shape()[1..]);
            assert_eq!(expected_strides, tensor.shape().strides());

            args.push(tensor.ptr().ptr());
        }

        let args = args.finish();

        let items = batch_size * self.inner_size;

        let blocks = ceil_div(items as u32, items_per_thread * threads_per_block);

        // TODO cache all of this so we just have to call launch_kernel at the end?
        self.function
            .launch_kernel(Dim3::single(blocks), Dim3::single(threads_per_block), 0, &stream, &args);
    }
}

fn build_operation(operand_types: &[String], operation: &str) -> String {
    let mut full_operation = String::new();
    let f = &mut full_operation;

    for (i, ty) in operand_types.iter().enumerate() {
        writeln!(
            f,
            "{ty} *x{i} = &(({ty} *) pointers[{i}])[offsets[{i}]];",
            ty = ty,
            i = i
        )
        .unwrap();
    }
    writeln!(f, "{}", operation).unwrap();

    full_operation
}
