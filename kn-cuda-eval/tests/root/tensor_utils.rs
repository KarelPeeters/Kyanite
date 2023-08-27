use itertools::Itertools;
use rand::Rng;

use kn_graph::dtype::{DTensor, IntoDScalar};
use kn_graph::ndarray::{ArcArray, Array1, Dimension, IntoDimension};

pub fn rng_tensor_f32<I: IntoDimension + Copy>(shape: I, rng: &mut impl Rng) -> DTensor {
    let size = shape.into_dimension().size();
    let data = rng_vec(size, rng);
    manual_tensor(shape, data)
}

pub fn manual_tensor<T: IntoDScalar, I: IntoDimension>(shape: I, data: Vec<T>) -> DTensor {
    T::vec_to_dtensor(data).reshape(shape)
}

pub fn linspace_tensor<I: IntoDimension + Copy>(shape: I) -> DTensor {
    let size = shape.into_dimension().size();
    let result = ArcArray::linspace(-1.0, 1.0, size).reshape(shape);
    DTensor::F32(result.into_dyn())
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
