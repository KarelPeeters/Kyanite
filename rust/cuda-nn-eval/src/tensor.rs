use itertools::Itertools;

use cuda_sys::wrapper::descriptor::{FilterDescriptor, TensorDescriptor};
use cuda_sys::wrapper::mem::DeviceMem;

use crate::shape::StridedShape;

#[derive(Debug)]
pub struct Tensor {
    pub mem: DeviceMem,
    pub shape: StridedShape,
}

impl Tensor {
    pub fn new(mem: DeviceMem, shape: StridedShape) -> Self {
        assert_eq!(shape.strided_size() * 4, mem.len(), "Buffer has wrong len for shape {:?}", shape);
        Tensor { mem, shape }
    }

    pub fn descriptor(&self) -> TensorDescriptor {
        let mut shape = self.shape.shape().iter().map(|&x| x as i32).collect_vec();
        let mut strides = self.shape.strides().iter().map(|&x| x as i32).collect_vec();

        // tensors descriptors and some cudnn operations seem to break with ranks < 4,
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

    pub fn filter_descriptor(&self) -> FilterDescriptor {
        assert_eq!(4, self.shape.rank(), "Filter must have rank 4");
        assert!(self.shape.has_simple_strides(), "Filter must have simple strides");

        let dims = self.shape.shape();
        FilterDescriptor::new(
            dims[0] as i32, dims[1] as i32, dims[2] as i32, dims[3] as i32,
        )
    }

    pub fn view(&self) -> Tensor {
        Tensor {
            mem: self.mem.view(),
            shape: self.shape.clone(),
        }
    }
}
