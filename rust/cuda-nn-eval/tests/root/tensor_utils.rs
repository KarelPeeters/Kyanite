use itertools::Itertools;

use nn_graph::cpu::Tensor;
use nn_graph::ndarray::{ArcArray, Dimension, IntoDimension};

pub fn manual_tensor<I: IntoDimension>(shape: I, data: Vec<f32>) -> Tensor {
    ArcArray::from_shape_vec(shape, data)
        .expect("Shape and data length mismatch")
        .into_dyn()
}

pub fn linspace_tensor<I: IntoDimension + Copy>(shape: I) -> ArcArray<f32, I::Dim> {
    let size = shape.into_dimension().size();
    ArcArray::linspace(-5.0, 5.0, size)
        .reshape(shape)
}

pub fn range_vec(len: usize) -> Vec<f32> {
    (0..len).map(|x| x as f32).collect_vec()
}