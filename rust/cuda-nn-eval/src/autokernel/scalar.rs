use itertools::zip_eq;
use std::fmt::Write;

use cuda_sys::wrapper::handle::{ComputeCapability, CudaStream};
use cuda_sys::wrapper::rtc::args::KernelArgs;
use cuda_sys::wrapper::rtc::core::{CuFunction, Dim3};

use crate::autokernel::{compile_cached_kernel, KernelKey};
use crate::device_tensor::DeviceTensor;
use crate::shape::StridedShape;

/// An instance of a scalar/elementwise kernel. Can be build for any operation, rank, strides, and sizes.
/// The first axis is runtime-dynamic without recompiling the kernel.
#[derive(Debug)]
pub struct ScalarKernel {
    capability: ComputeCapability,
    function: CuFunction,

    inner_size: usize,
    inner_shape: Vec<usize>,
    operand_types: Vec<String>,
    operand_strides: Vec<Vec<isize>>,
}

const TEMPLATE_SOURCE: &str = include_str!("scalar.cu");

impl ScalarKernel {
    pub fn new(
        capability: ComputeCapability,
        inner_shape: Vec<usize>,
        operand_types: Vec<String>,
        operand_strides: Vec<Vec<isize>>,
        operation: &str,
    ) -> Self {
        assert!(operand_types.len() > 0);
        assert_eq!(operand_strides.len(), operand_types.len());
        assert!(
            operation.ends_with(";"),
            "Operation should end with ';', got {:?}",
            operation
        );

        let full_operation = build_operation(&operand_types, operation);
        let replacements = build_replacements(&inner_shape, &operand_strides, &full_operation);

        let source = replacements
            .iter()
            .fold(TEMPLATE_SOURCE.to_owned(), |source, (key, value)| {
                assert!(source.contains(key), "Source does not contain key {}", key);
                source.replace(key, value)
            });
        assert!(
            !source.contains('$'),
            "Source still contains '$', probably failed to replace all parameters"
        );

        let key = KernelKey {
            capability,
            source,
            func_name: "scalar_kernel".to_owned(),
        };

        let function = compile_cached_kernel(key);
        let inner_size = inner_shape.iter().product();

        ScalarKernel {
            capability,
            function,
            inner_size,
            inner_shape,
            operand_types,
            operand_strides,
        }
    }

    pub unsafe fn run(&self, stream: &CudaStream, tensors: &[DeviceTensor]) {
        assert_eq!(stream.device().compute_capability(), self.capability);
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

        let items_per_thread = 1024;
        let threads_per_block = 128;
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

fn build_replacements(
    inner_shape: &[usize],
    operand_strides: &[Vec<isize>],
    operation: &str,
) -> Vec<(&'static str, String)> {
    let mut full_shape = vec![0];
    full_shape.extend_from_slice(inner_shape);

    let dense = StridedShape::new_simple(full_shape.to_vec());

    let mut dense_strides = String::from("{");
    append_int_array(&mut dense_strides, dense.strides());
    dense_strides.push('}');

    let mut strides = String::from("{");
    for (i, op_stride) in operand_strides.iter().enumerate() {
        assert_eq!(op_stride.len(), dense.rank());
        if i != 0 {
            strides.push_str(", ");
        }
        append_int_array(&mut strides, op_stride);
    }
    strides.push('}');

    vec![
        ("$RANK$", format!("{}", dense.rank())),
        ("$OPERANDS$", format!("{}", operand_strides.len())),
        ("$STRIDES_DENSE$", dense_strides),
        ("$STRIDES$", strides),
        ("$OPERATION$", operation.to_owned()),
    ]
}

fn append_int_array(s: &mut String, values: &[isize]) {
    for (i, v) in values.iter().enumerate() {
        if i != 0 {
            s.push_str(", ");
        }
        write!(s, "{}", v).unwrap();
    }
}

fn ceil_div(x: u32, y: u32) -> u32 {
    (x + y - 1) / y
}
