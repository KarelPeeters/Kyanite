use rand::rngs::StdRng;
use rand::SeedableRng;

use kn_graph::dtype::DType;
use kn_graph::graph::{BinaryOp, Graph, Operation, SliceRange, UnaryOp};
use kn_graph::optimizer::{optimize_graph, OptimizerSettings};
use kn_graph::shape;
use kn_graph::shape::Shape;

use crate::root::runner::test_all;
use crate::root::tensor_utils::{manual_tensor, rng_tensor_f32, rng_vec};

#[test]
fn chained_scalar() {
    let mut graph = Graph::new();

    let input = graph.input(shape![2, 3], DType::F32);

    let w1 = graph.constant::<f32>(shape![1, 3], vec![1.0, 2.0, 3.0]);
    let w2 = graph.constant::<f32>(shape![2, 1], vec![0.1, 0.2]);

    let y1 = graph.mul(input, w1);
    let y2 = graph.add(y1, w2);

    graph.output(y2);

    let input_tensor = manual_tensor::<f32, _>((2, 3), vec![1.0; 6]);
    let output_tensor = manual_tensor::<f32, _>((2, 3), vec![1.1, 2.1, 3.1, 1.2, 2.2, 3.2]);

    test_all(&graph, 0, &[input_tensor], Some(&[output_tensor]))
}

#[test]
fn split_scalar() {
    let mut graph = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);

    let input = graph.input(shape![2, 3], DType::F32);

    let w1 = graph.constant::<f32>(shape![1, 3], rng_vec(3, &mut rng));
    let w2 = graph.constant::<f32>(shape![2, 1], rng_vec(2, &mut rng));

    let y1 = graph.mul(input, w1);
    let y2 = graph.add(input, w2);

    graph.output_all(&[y1, y2]);

    let input_tensor = rng_tensor_f32((2, 3), &mut rng);
    test_all(&graph, 0, &[input_tensor], None)
}

#[test]
fn rejoining_scalar() {
    let mut graph = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);

    let input = graph.input(shape![2, 3], DType::F32);

    let w1 = graph.constant::<f32>(shape![1, 3], rng_vec(3, &mut rng));
    let w2 = graph.constant::<f32>(shape![2, 1], rng_vec(2, &mut rng));

    let y1 = graph.mul(input, w1);
    let y2 = graph.add(input, w2);
    let y3 = graph.binary(BinaryOp::Max, y1, y2);

    graph.output(y3);

    let input_tensor = rng_tensor_f32((2, 3), &mut rng);
    test_all(&graph, 0, &[input_tensor], None)
}

#[test]
fn inner_scalar_used() {
    let mut graph = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);

    let input = graph.input(shape![2, 3], DType::F32);

    let w1 = graph.constant::<f32>(shape![1, 3], rng_vec(3, &mut rng));
    let w2 = graph.constant::<f32>(shape![2, 1], rng_vec(2, &mut rng));

    let y1 = graph.mul(input, w1);
    let y2 = graph.add(input, w2);
    let y3 = graph.binary(BinaryOp::Max, y1, y2);

    graph.output_all(&[y1, y3]);

    let input_tensor = rng_tensor_f32((2, 3), &mut rng);
    test_all(&graph, 0, &[input_tensor], None)
}

#[test]
fn gather_cast() {
    let mut graph = Graph::new();

    let x = graph.input(shape![4], DType::F32);

    let i_float = graph.constant(shape![1], vec![0f32]);
    let i = graph.unary(UnaryOp::ValueCast(DType::U32), i_float);

    let y = graph.gather(x, 0, i);
    graph.output(y);

    println!("Raw graph:");
    println!("{}", graph);

    let opt_graph = optimize_graph(&graph, OptimizerSettings::default());
    println!("Optimized graph:");
    println!("{}", opt_graph);

    let opt_x = opt_graph.inputs()[0];
    let opt_y = opt_graph.outputs()[0];

    // make sure that everything collapses to a single slice operation
    // (either during graph construction or during optimization, that's not important)
    let expected = Operation::Slice {
        input: opt_x,
        axis: 0,
        range: SliceRange::single(0),
    };
    assert_eq!(opt_graph[opt_y].operation, expected);
}

#[test]
fn cast_id() {
    let mut graph = Graph::new();

    let dtype = DType::F32;

    let x = graph.input(Shape::SCALAR, dtype);
    let y0 = graph.unary(UnaryOp::BitCast(dtype), x);
    let y1 = graph.unary(UnaryOp::ValueCast(dtype), x);

    assert_eq!(y0, x);
    assert_eq!(y1, x);
}
