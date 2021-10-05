use itertools::{Itertools, zip_eq};

use cuda_sys::wrapper::descriptor::{FilterDescriptor, TensorDescriptor};
use cuda_sys::wrapper::mem::DeviceMem;
use nn_graph::shape::ConcreteShape;

#[derive(Debug)]
pub struct Tensor {
    pub mem: DeviceMem,
    pub shape: ConcreteShape,
    pub strides: Vec<usize>,
    pub has_basic_strides: bool,
}

impl Tensor {
    pub fn new_basic(mem: DeviceMem, shape: ConcreteShape) -> Self {
        assert_eq!(shape.size() * 4, mem.len_bytes(), "Buffer has wrong len for shape {:?}", shape);
        Tensor {
            mem,
            has_basic_strides: true,
            strides: basic_strides(&shape),
            shape,
        }
    }

    pub fn new_special(mem: DeviceMem, shape: ConcreteShape, strides: Vec<usize>) -> Self {
        assert_eq!(
            len_from_shape_stride(&shape, &strides) * 4, mem.len_bytes(),
            "Buffer has wrong len for shape {:?}; strides {:?}",
            shape, strides,
        );
        Tensor {
            mem,
            has_basic_strides: strides == basic_strides(&shape),
            strides,
            shape,
        }
    }

    pub fn descriptor(&self) -> TensorDescriptor {
        let mut shape = self.shape.dims.iter().map(|&d| d as i32).collect_vec();
        let mut strides = self.strides.iter().map(|&d| d as i32).collect_vec();

        // tensors themselves and other cudnn operations seem to break with ranks < 4,
        //   so pad the rank until it's large enough
        while shape.len() < 4 {
            shape.push(1);
            strides.push(1);
        }

        TensorDescriptor::new(
            shape,
            strides,
        )
    }

    pub fn descriptor_filter(&self) -> FilterDescriptor {
        assert_eq!(4, self.shape.rank());
        assert!(self.has_basic_strides);

        let dims = &self.shape.dims;
        FilterDescriptor::new(
            dims[0] as i32, dims[1] as i32, dims[2] as i32, dims[3] as i32,
        )
    }

    pub fn view(&self) -> Tensor {
        Tensor {
            mem: self.mem.view(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            has_basic_strides: self.has_basic_strides,
        }
    }
}

fn basic_strides(shape: &ConcreteShape) -> Vec<usize> {
    let mut curr = 1;
    let mut result = vec![];

    for &d in shape.dims.iter().rev() {
        result.push(curr);
        curr *= d;
    }

    result.reverse();
    result
}

pub fn len_from_shape_stride(shape: &ConcreteShape, strides: &[usize]) -> usize {
    //TODO what about size == 0?
    let max_index = zip_eq(&shape.dims, strides)
        .map(|(&size, &stride)| {
            assert_ne!(size, 0);
            stride * (size - 1)
        })
        .sum::<usize>();

    let len = max_index + 1;
    len
}

