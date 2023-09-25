use itertools::Itertools;
use rand::Rng;

use kn_graph::cpu::Tensor;
use kn_graph::ndarray::{ArcArray, Array1, Dimension, IntoDimension};

pub fn rng_tensor<I: IntoDimension + Copy>(shape: I, rng: &mut impl Rng) -> Tensor {
    let size = shape.into_dimension().size();
    let data = rng_vec(size, rng);
    manual_tensor(shape, data)
}

pub fn manual_tensor<I: IntoDimension>(shape: I, data: Vec<f32>) -> Tensor {
    ArcArray::from_shape_vec(shape, data)
        .expect("Shape and data length mismatch")
        .into_dyn()
}

pub fn linspace_tensor<I: IntoDimension + Copy>(shape: I) -> ArcArray<f32, I::Dim> {
    let size = shape.into_dimension().size();
    ArcArray::linspace(-1.0, 1.0, size).reshape(shape)
}

pub fn rng_vec(len: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..len).map(|_| rng.gen()).collect_vec()
}

pub fn range_vec(len: usize) -> Vec<f32> {
    (0..len).map(|x| x as f32).collect_vec()
}

pub fn linspace_vec(len: usize) -> Vec<f32> {
    Array1::linspace(-1.0, 1.0, len).to_vec()
}
