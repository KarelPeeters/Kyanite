use bytemuck::{cast_slice, cast_slice_mut};
use itertools::Itertools;

use cuda_sys::wrapper::descriptor::{FilterDescriptor, TensorDescriptor};
use cuda_sys::wrapper::mem::device::DeviceMem;

use crate::shape::StridedShape;

#[derive(Debug)]
pub struct Tensor {
    pub mem: DeviceMem,
    pub shape: StridedShape,
}

impl Tensor {
    pub fn new(mem: DeviceMem, shape: StridedShape) -> Self {
        assert_eq!(
            shape.strided_size() * 4,
            mem.len_bytes(),
            "Buffer has wrong len for shape {:?}",
            shape
        );
        Tensor { mem, shape }
    }

    pub fn descriptor(&self) -> TensorDescriptor {
        let mut shape = self.shape.shape().iter().map(|&x| x as i32).collect_vec();
        let mut strides = self.shape.strides().iter().map(|&x| x as i32).collect_vec();

        // tensor descriptors and some cudnn operations seem to break with ranks < 4,
        //   so pad the rank until it's large enough
        while shape.len() < 4 {
            shape.push(1);
            strides.push(1);
        }

        TensorDescriptor::new(shape, strides)
    }

    pub fn filter_descriptor(&self) -> FilterDescriptor {
        assert_eq!(4, self.shape.rank(), "Filter must have rank 4");
        assert!(self.shape.has_simple_strides(), "Filter must have simple strides");

        let dims = self.shape.shape();
        FilterDescriptor::new(dims[0] as i32, dims[1] as i32, dims[2] as i32, dims[3] as i32)
    }

    /// Returns a (shallow) clone of this tensor, pointing to the same memory.
    pub fn view(&self) -> Tensor {
        Tensor {
            mem: self.mem.view(),
            shape: self.shape.clone(),
        }
    }

    pub fn permute(&self, permutation: &[usize]) -> Tensor {
        Tensor::new(self.mem.view(), self.shape.permute(permutation))
    }

    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Tensor {
        // Steps to slice a tensor:
        //  * use the new shape
        //  * keep the old strides
        //  * offset initial pointer to account for `start`
        //  * limit the buffer length based on the new size
        let result_shape = self.shape.slice(axis, start, end);

        let start_bytes = result_shape.strides()[axis] * start * 4;
        let len_bytes = result_shape.strided_size() * 4;

        let mem = self.mem.slice_bytes(start_bytes, len_bytes);
        Tensor::new(mem, result_shape)
    }

    pub unsafe fn copy_from(&self, other: &Tensor) {
        assert_eq!(
            self.shape, other.shape,
            "Both tensors must have the same shape and stride for now"
        );
        self.mem.copy_from_device(&other.mem);
    }

    pub unsafe fn copy_from_host(&self, buffer: &[f32]) {
        assert!(
            self.shape.has_simple_strides(),
            "Tensor must have simple stride for now, got {:?}",
            self.shape
        );
        self.mem.copy_from_host(cast_slice(buffer));
    }

    pub unsafe fn copy_to_host(&self, buffer: &mut [f32]) {
        assert!(
            self.shape.has_simple_strides(),
            "Tensor must have simple stride for now, got {:?}",
            self.shape
        );
        self.mem.copy_to_host(cast_slice_mut(buffer));
    }
}
