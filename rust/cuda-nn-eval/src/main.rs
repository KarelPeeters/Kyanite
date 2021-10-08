use nn_graph::ndarray::Array;

fn main() {
    let tensor = Array::<f32, _>::zeros((12, 4));
    println!("{:?}", tensor.shape());
    println!("{:?}", tensor.strides());

    let r = tensor.into_shape((4, 6, 2))
        .unwrap();

    println!("{:?}", r.shape());
    println!("{:?}", r.strides());
}