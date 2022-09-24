use crate::root::runner::test_all;
use crate::root::tensor_utils::{manual_tensor, rng_tensor, rng_vec};
use nn_graph::graph::{BinaryOp, Graph};
use nn_graph::shape;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn chained_scalar() {
    let mut graph = Graph::new();

    let input = graph.input(shape![2, 3]);

    let w1 = graph.constant(shape![1, 3], vec![1.0, 2.0, 3.0]);
    let w2 = graph.constant(shape![2, 1], vec![0.1, 0.2]);

    let y1 = graph.mul(input, w1);
    let y2 = graph.add(y1, w2);

    graph.output(y2);

    let input_tensor = manual_tensor((2, 3), vec![1.0; 6]);
    let output_tensor = manual_tensor((2, 3), vec![1.1, 2.1, 3.1, 1.2, 2.2, 3.2]);

    test_all(&graph, 0, &[input_tensor], Some(&[output_tensor]))
}

#[test]
fn split_scalar() {
    let mut graph = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);

    let input = graph.input(shape![2, 3]);

    let w1 = graph.constant(shape![1, 3], rng_vec(3, &mut rng));
    let w2 = graph.constant(shape![2, 1], rng_vec(2, &mut rng));

    let y1 = graph.mul(input, w1);
    let y2 = graph.add(input, w2);

    graph.output_all(&[y1, y2]);

    let input_tensor = rng_tensor((2, 3), &mut rng);
    test_all(&graph, 0, &[input_tensor], None)
}

#[test]
fn rejoining_scalar() {
    let mut graph = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);

    let input = graph.input(shape![2, 3]);

    let w1 = graph.constant(shape![1, 3], rng_vec(3, &mut rng));
    let w2 = graph.constant(shape![2, 1], rng_vec(2, &mut rng));

    let y1 = graph.mul(input, w1);
    let y2 = graph.add(input, w2);
    let y3 = graph.binary(BinaryOp::Max, y1, y2);

    graph.output(y3);

    let input_tensor = rng_tensor((2, 3), &mut rng);
    test_all(&graph, 0, &[input_tensor], None)
}

#[test]
fn inner_scalar_used() {
    let mut graph = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);

    let input = graph.input(shape![2, 3]);

    let w1 = graph.constant(shape![1, 3], rng_vec(3, &mut rng));
    let w2 = graph.constant(shape![2, 1], rng_vec(2, &mut rng));

    let y1 = graph.mul(input, w1);
    let y2 = graph.add(input, w2);
    let y3 = graph.binary(BinaryOp::Max, y1, y2);

    graph.output_all(&[y1, y3]);

    let input_tensor = rng_tensor((2, 3), &mut rng);
    test_all(&graph, 0, &[input_tensor], None)
}
