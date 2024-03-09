use decorum::Total;
use itertools::Itertools;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use kn_graph::dtype::{DTensor, DType};
use kn_graph::graph::{BinaryOp, Graph, Operation, ReduceOp, SliceRange, UnaryOp, Value};
use kn_graph::ndarray::{Array, Array1};
use kn_graph::optimizer::optimize_graph;
use kn_graph::shape;
use kn_graph::shape::{Shape, Size};

use crate::root::runner::test_all;
use crate::root::tensor_utils::{linspace_tensor, linspace_vec, manual_tensor, range_vec, rng_tensor_f32, rng_vec};

#[test]
fn empty() {
    test_all(&Graph::new(), 8, &[], Some(&[]))
}

#[test]
fn copy() {
    let mut graph = Graph::new();

    let fixed_size = 10;
    let batch_size = 8;

    let fixed = graph.input(shape![fixed_size], DType::F32);
    let batch = graph.input(shape![Size::BATCH], DType::F32);
    graph.output_all(&[fixed, batch]);

    let fixed_tensor = linspace_tensor(fixed_size);
    let batch_tensor = linspace_tensor(batch_size);

    test_all(
        &graph,
        batch_size,
        &[fixed_tensor.clone(), batch_tensor.clone()],
        Some(&[fixed_tensor, batch_tensor]),
    )
}

#[test]
fn slice() {
    let mut graph = Graph::new();

    let input = graph.input(shape![10, 4], DType::F32);
    let input_tensor = linspace_tensor((10, 4));

    let indexed = graph.index(input, 1, 0);
    let outputs = [
        // start:end slicing
        indexed,
        graph.slice(input, 0, SliceRange::new(0, 2, 1)),
        graph.slice(indexed, 0, SliceRange::new(0, 2, 1)),
    ];

    graph.output_all(&outputs);

    test_all(&graph, 0, &[input_tensor], None)
}

#[test]
fn flip() {
    let mut graph = Graph::new();

    let x = graph.constant::<f32>(shape![2, 3].clone(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let first_once = graph.flip(x, 0);
    let first_twice = graph.flip(first_once, 0);

    let other_once = graph.flip(x, 1);
    let other_twice = graph.flip(other_once, 1);

    let combined = graph.flip(first_once, 1);

    graph.output_all(&[first_once, first_twice, other_once, other_twice, combined]);

    test_all(
        &graph,
        0,
        &[],
        Some(&[
            manual_tensor::<f32, _>((2, 3), vec![3.0, 4.0, 5.0, 0.0, 1.0, 2.0]),
            manual_tensor::<f32, _>((2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            manual_tensor::<f32, _>((2, 3), vec![2.0, 1.0, 0.0, 5.0, 4.0, 3.0]),
            manual_tensor::<f32, _>((2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            manual_tensor::<f32, _>((2, 3), vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0]),
        ]),
    );
}

#[test]
fn flip_conv() {
    let mut graph = Graph::new();

    let input = graph.input(shape![2, 4, 8, 8], DType::F32);
    let flipped = graph.flip(input, 3);

    let weight = graph.constant::<f32>(shape![4, 4, 3, 3], linspace_vec(4 * 4 * 3 * 3));
    let result = graph.conv(flipped, weight, 1, 1, 1, 1);

    graph.output(result);
    test_all(&graph, 0, &[linspace_tensor((2, 4, 8, 8))], None);
}

#[test]
fn repeat() {
    let mut graph = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);
    let input = graph.input(shape![2, 3], DType::F32);

    let outputs = [
        graph.repeat(input, 0, Size::fixed(0)),
        graph.repeat(input, 0, Size::fixed(2)),
        graph.repeat(input, 1, Size::fixed(0)),
        graph.repeat(input, 1, Size::fixed(2)),
    ];
    graph.output_all(&outputs);

    test_all(&graph, 0, &[rng_tensor_f32((2, 3), &mut rng)], None);
}

#[test]
fn repeat_interleave() {
    let mut graph = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);
    let input = graph.input(shape![2, 3], DType::F32);

    let outputs = [
        graph.repeat_interleave(input, 0, Size::fixed(0)),
        graph.repeat_interleave(input, 0, Size::fixed(2)),
        graph.repeat_interleave(input, 1, Size::fixed(0)),
        graph.repeat_interleave(input, 1, Size::fixed(2)),
    ];
    graph.output_all(&outputs);

    test_all(&graph, 0, &[rng_tensor_f32((2, 3), &mut rng)], None);
}

#[test]
fn repeat_manual() {
    let mut graph = Graph::new();

    let input = graph.input(shape![3], DType::F32);
    let outputs = [
        graph.repeat(input, 0, Size::fixed(2)),
        graph.repeat_interleave(input, 0, Size::fixed(2)),
    ];
    graph.output_all(&outputs);

    let input_tensor = manual_tensor::<f32, _>((3,), vec![1.0, 2.0, 3.0]);
    let output_tensors = [
        manual_tensor::<f32, _>((6,), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
        manual_tensor::<f32, _>((6,), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
    ];

    test_all(&graph, 0, &[input_tensor], Some(&output_tensors))
}

#[test]
fn gather_simple_axis_0() {
    let mut graph = Graph::new();

    let input = graph.constant::<f32>(shape![2, 3], vec![7.0, 7.1, 7.2, 7.3, 7.4, 7.5]);
    let index = graph.constant::<u32>(shape![4], vec![0, 1, 1, 0]);
    let output = graph.gather(input, 0, index);
    let output_tensor =
        manual_tensor::<f32, _>((4, 3), vec![7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.3, 7.4, 7.5, 7.0, 7.1, 7.2]);
    graph.output(output);

    test_all(&graph, 0, &[], Some(&[output_tensor]))
}

#[test]
fn gather_simple_axis_1() {
    let mut graph = Graph::new();

    let input = graph.constant::<f32>(shape![2, 3], vec![7.0, 7.1, 7.2, 7.3, 7.4, 7.5]);
    let index = graph.constant::<u32>(shape![4], vec![0, 2, 1, 0]);
    let output = graph.gather(input, 1, index);
    let output_tensor = manual_tensor::<f32, _>((2, 4), vec![7.0, 7.2, 7.1, 7.0, 7.3, 7.5, 7.4, 7.3]);
    graph.output(output);

    test_all(&graph, 0, &[], Some(&[output_tensor]))
}

#[test]
fn gather_complex_axis_0() {
    let mut graph = Graph::new();

    let input = graph.input(shape![3, 2], DType::F32);
    let input_tensor = manual_tensor::<f32, _>((3, 2), vec![1.0, 1.2, 2.3, 3.4, 4.5, 5.7]);
    let indices = graph.constant::<u32>(shape![2, 2], vec![0, 1, 1, 2]);
    let output = graph.gather(input, 0, indices);
    let output_tensor = manual_tensor::<f32, _>((2, 2, 2), vec![1.0, 1.2, 2.3, 3.4, 2.3, 3.4, 4.5, 5.7]);
    graph.output(output);

    test_all(&graph, 0, &[input_tensor], Some(&[output_tensor]));
}

#[test]
fn gather_complex_axis_1() {
    let mut graph = Graph::new();

    let input = graph.input(shape![3, 3], DType::F32);
    let input_tensor = manual_tensor::<f32, _>((3, 3), vec![1.0, 1.2, 1.9, 2.3, 3.4, 3.9, 4.5, 5.7, 5.9]);
    let indices = graph.constant::<u32>(shape![1, 2], vec![0, 2]);
    let output = graph.gather(input, 1, indices);
    let output_tensor = manual_tensor::<f32, _>((3, 1, 2), vec![1.0, 1.9, 2.3, 3.9, 4.5, 5.9]);
    graph.output(output);

    test_all(&graph, 0, &[input_tensor], Some(&[output_tensor]));
}

#[test]
fn gather_size_0() {
    let mut graph = Graph::new();

    let input = graph.input(shape![8, 4], DType::F32);
    let input_tensor = linspace_tensor((8, 4));

    let indices = graph.constant::<u32>(shape![0], vec![]);
    let output0 = graph.gather(input, 0, indices);
    let output1 = graph.gather(input, 1, indices);
    graph.output(output0);
    graph.output(output1);

    assert_eq!(graph[output0].shape, shape![0, 4]);
    assert_eq!(graph[output1].shape, shape![8, 0]);

    test_all(&graph, 0, &[input_tensor], None);
}

#[test]
fn gather_as_index() {
    let mut graph = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);

    let input = graph.input(shape![16, 8, 4], DType::F32);
    let index = graph.scalar::<u32>(4);
    let indices = graph.view(index, shape![1]);
    let indices_repeat = graph.broadcast(indices, shape![3]);

    let result = graph.gather(input, 1, indices);
    let result_repeat = graph.gather(input, 1, indices_repeat);

    graph.output_all(&[result, result_repeat]);

    // test that this gather is actually implemented as a simple index operation
    println!("{}", graph);
    for v in graph.values() {
        assert!(
            !matches!(graph[v].operation, Operation::Gather { .. }),
            "Graph contains a gather operation, it should have been replaced by an index operation"
        );
    }

    test_all(&graph, 0, &[rng_tensor_f32((16, 8, 4), &mut rng)], None);
}

#[test]
fn linear() {
    let mut graph = Graph::new();

    let input = graph.input(shape![1, 4], DType::F32);
    let weight = graph.constant::<f32>(shape![2, 4], range_vec(8));
    let bias = graph.constant::<f32>(shape![1, 2], vec![-10.0, 10.0]);

    let linear = graph.linear(input, weight);
    let biased = graph.add(linear, bias);

    graph.output_all(&[linear, biased]);

    test_all(
        &graph,
        0,
        &[manual_tensor::<f32, _>((1, 4), vec![0.0, 1.0, 2.0, 3.0])],
        Some(&[
            manual_tensor::<f32, _>((1, 2), vec![14.0, 38.0]),
            manual_tensor::<f32, _>((1, 2), vec![4.0, 48.0]),
        ]),
    )
}

#[test]
fn linear_sliced() {
    let mut graph = Graph::new();

    let left = graph.input(shape![8, 4], DType::F32);
    let left_sliced = graph.slice(left, 0, SliceRange::new(0, 8, 2));
    let right = graph.input(shape![4, 3], DType::F32);

    let result = graph.mat_mul(left_sliced, right);
    graph.output(result);

    test_all(&graph, 0, &[linspace_tensor((8, 4)), linspace_tensor((4, 3))], None);
}

#[test]
fn mat_mul_broadcast() {
    let shapes = vec![
        (shape![4, 2], shape![2, 3], shape![4, 3]),
        (shape![8, 4, 2], shape![2, 3], shape![8, 4, 3]),
        (shape![4, 2], shape![8, 2, 3], shape![8, 4, 3]),
        (shape![8, 4, 2], shape![8, 2, 3], shape![8, 4, 3]),
        (shape![1, 1, 1, 4, 2], shape![8, 1, 2, 3], shape![1, 8, 1, 4, 3]),
    ];

    let mut rng = StdRng::seed_from_u64(0);
    let mut graph = Graph::new();

    let mut inputs = vec![];

    for (left_shape, right_shape, result_shape) in shapes {
        println!("Testing {} @ {} => {}", left_shape, right_shape, result_shape);

        inputs.push(rng_tensor_f32(
            left_shape.unwrap_fixed("shape").dims.as_slice(),
            &mut rng,
        ));
        inputs.push(rng_tensor_f32(
            right_shape.unwrap_fixed("shape").dims.as_slice(),
            &mut rng,
        ));

        let left = graph.input(left_shape.clone(), DType::F32);
        let right = graph.input(right_shape.clone(), DType::F32);
        let result = graph.mat_mul(left, right);

        assert_eq!(graph[result].shape, result_shape);
        graph.output(result);
    }

    test_all(&graph, 0, &inputs, None);
}

#[test]
fn mat_mul_transpose() {
    // run the "transposed" cases first since they're simpler for cublas
    for transpose_a in [true, false] {
        for transpose_b in [true, false] {
            println!("Transpose a: {}, b: {}", transpose_a, transpose_b);
            let mut graph = Graph::new();

            let mut shape_a = shape![4, 5, 6];
            let mut shape_b = shape![4, 6, 3];

            if transpose_a {
                shape_a.dims.swap(1, 2);
            }
            if transpose_b {
                shape_b.dims.swap(1, 2);
            }

            let a_orig = graph.constant::<f32>(shape_a, linspace_vec(4 * 5 * 6));
            let b_orig = graph.constant::<f32>(shape_b, linspace_vec(4 * 6 * 3));

            let a = if transpose_a {
                graph.permute(a_orig, vec![0, 2, 1])
            } else {
                a_orig
            };
            let b = if transpose_b {
                graph.permute(b_orig, vec![0, 2, 1])
            } else {
                b_orig
            };

            let result = graph.batched_mat_mul(a, b);
            assert_eq!(graph[result].shape, shape![4, 5, 3]);
            graph.output(result);

            test_all(&graph, 0, &[], None);
        }
    }
}

#[test]
fn horizontal_1x1_conv() {
    let mut graph = Graph::new();

    let input = graph.constant::<f32>(shape![2, 4, 1, 8], linspace_vec(2 * 4 * 8));
    let filter = graph.constant::<f32>(shape![3, 4, 1, 1], linspace_vec(3 * 4));

    let output = graph.conv(input, filter, 1, 1, 0, 0);
    graph.output(output);

    assert_eq!(graph[output].shape, shape![2, 3, 1, 8]);
    test_all(&graph, 0, &[], None)
}

#[test]
fn vertical_1x1_conv() {
    let mut graph = Graph::new();

    let input = graph.constant::<f32>(shape![2, 4, 8, 1], linspace_vec(2 * 4 * 8));
    let filter = graph.constant::<f32>(shape![3, 4, 1, 1], linspace_vec(3 * 4));

    let output = graph.conv(input, filter, 1, 1, 0, 0);
    graph.output(output);

    assert_eq!(graph[output].shape, shape![2, 3, 8, 1]);
    test_all(&graph, 0, &[], None)
}

#[test]
fn fuse_clamp() {
    let mut graph = Graph::new();

    let mut curr = graph.input(shape![Size::BATCH], DType::F32);

    curr = graph.clamp::<f32>(curr, -5.0, f32::INFINITY);
    curr = graph.clamp::<f32>(curr, f32::NEG_INFINITY, 2.0);
    curr = graph.clamp::<f32>(curr, 0.0, 1.0);
    curr = graph.clamp::<f32>(curr, -1.0, 2.0);

    graph.output(curr);

    test_all(
        &graph,
        5,
        &[manual_tensor::<f32, _>(5, vec![-2.0, 0.0, 0.5, 1.0, 2.0])],
        Some(&[manual_tensor::<f32, _>(5, vec![0.0, 0.0, 0.5, 1.0, 1.0])]),
    )
}

#[test]
fn ele_broadcast() {
    // don't test division, since the GPU doesn't support it yet
    for op in [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Min,
        BinaryOp::Max,
    ] {
        println!("Testing operation {:?}", op);

        let mut graph = Graph::new();
        let left = graph.constant::<f32>(shape![2, 3, 4], linspace_vec(2 * 3 * 4));

        for shape in [Shape::SCALAR, shape![1, 1, 1], shape![2, 3, 4], shape![2, 1, 4]] {
            println!("  with right shape {}", shape);
            let size = shape.size().eval(0);
            let right = graph.constant::<f32>(shape, linspace_vec(size));
            let result = graph.binary(op, left, right);
            graph.output(result);
        }

        test_all(&graph, 0, &[], None);
    }
}

#[test]
fn add_broadcast() {
    let mut graph = Graph::new();

    let left = graph.constant::<f32>(shape![2, 2, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let right = graph.constant::<f32>(shape![1, 2, 2], vec![0.0, 1.0, 2.0, 3.0]);
    let scalar = graph.constant::<f32>(shape![], vec![10.0]);

    let output0 = graph.add(left, right);
    let output1 = graph.add(right, left);
    let output2 = graph.add(right, scalar);

    graph.output_all(&[output0, output1, output2]);

    let expected_output_01 = manual_tensor::<f32, _>((2, 2, 2), vec![0.0, 2.0, 4.0, 6.0, 4.0, 6.0, 8.0, 10.0]);
    let expected_output2 = manual_tensor::<f32, _>((1, 2, 2), vec![10.0, 11.0, 12.0, 13.0]);
    let expected_outputs = [expected_output_01.clone(), expected_output_01, expected_output2];

    test_all(&graph, 0, &[], Some(&expected_outputs))
}

#[test]
fn affine_single_element() {
    let input_data = manual_tensor::<f32, _>((8, 1, 1, 1), range_vec(8));
    let output_data = input_data
        .unwrap_f32()
        .unwrap()
        .map(|&x| ((x + 1.0) * 2.0 * 10.0 + 3.0) * 4.0)
        .to_shared();

    let mut graph = Graph::new();

    let const_shape = shape![1, 1, 1, 1];
    let bias_0 = graph.constant::<f32>(const_shape.clone(), vec![1.0]);
    let scale_0 = graph.constant::<f32>(const_shape.clone(), vec![2.0]);
    let filter = graph.constant::<f32>(const_shape.clone(), vec![10.0]);
    let bias_1 = graph.constant::<f32>(const_shape.clone(), vec![3.0]);
    let scale_1 = graph.constant::<f32>(const_shape.clone(), vec![4.0]);

    let curr = graph.input(Shape::fixed(input_data.shape()), DType::F32);
    let curr = graph.add(curr, bias_0);
    let curr = graph.mul(curr, scale_0);
    let curr = graph.conv(curr, filter, 1, 1, 0, 0);
    let curr = graph.add(curr, bias_1);
    let curr = graph.mul(curr, scale_1);
    graph.output(curr);

    test_all(&graph, 0, &[input_data], Some(&[DTensor::F32(output_data)]))
}

#[test]
fn affine_add_twice() {
    let mut graph = Graph::new();

    let x = graph.input(shape![Size::BATCH, 1, 1, 1], DType::F32);
    let w1 = graph.constant::<f32>(shape![1, 1, 1, 1], vec![1.0]);
    let w2 = graph.constant::<f32>(shape![1, 1, 1, 1], vec![2.0]);

    let y1 = graph.add(x, w1);
    let y2 = graph.add(y1, w2);

    graph.output(y2);

    test_all(
        &graph,
        2,
        &[manual_tensor::<f32, _>((2, 1, 1, 1), vec![0.0, 1.0])],
        Some(&[manual_tensor::<f32, _>((2, 1, 1, 1), vec![3.0, 4.0])]),
    )
}

#[test]
fn affine_single_div() {
    let mut graph = Graph::new();

    let left = graph.constant::<f32>(shape![2, 3], range_vec(2 * 3));
    let right = graph.scalar::<f32>(2.0);
    let result = graph.binary(BinaryOp::Div, left, right);
    graph.output(result);

    let expected = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
    test_all(&graph, 0, &[], Some(&[manual_tensor::<f32, _>((2, 3), expected)]));
}

#[test]
fn affine_multiple_channels() {
    let input_data = manual_tensor::<f32, _>((8, 3, 1, 1), range_vec(8 * 3));

    let mut graph = Graph::new();

    let before_shape = shape![1, 3, 1, 1];
    let after_shape = shape![1, 2, 1, 1];
    let filter_shape = shape![2, 3, 1, 1];

    let bias_0 = graph.constant::<f32>(before_shape.clone(), vec![1.0, 2.0, 3.0]);
    let scale_0 = graph.constant::<f32>(before_shape.clone(), vec![2.0, 3.0, 4.0]);
    let filter = graph.constant::<f32>(filter_shape.clone(), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let bias_1 = graph.constant::<f32>(after_shape.clone(), vec![3.0, 4.0]);
    let scale_1 = graph.constant::<f32>(after_shape.clone(), vec![4.0, 5.0]);

    let curr = graph.input(Shape::fixed(input_data.shape()), DType::F32);
    let curr = graph.add(curr, bias_0);
    let curr = graph.mul(curr, scale_0);
    let curr = graph.conv(curr, filter, 1, 1, 0, 0);
    let curr = graph.add(curr, bias_1);
    let curr = graph.mul(curr, scale_1);
    graph.output(curr);

    test_all(&graph, 0, &[input_data], None)
}

#[test]
fn conv_padding() {
    let mut graph = Graph::new();

    let input0 = graph.input(shape![1, 1, 1, 1], DType::F32);
    let input1 = graph.input(shape![1, 1, 8, 8], DType::F32);

    let filter = graph.constant::<f32>(shape![1, 1, 3, 3], range_vec(3 * 3));

    let output00 = graph.conv(input0, filter, 1, 1, 1, 1);
    let output01 = graph.conv(input0, filter, 1, 1, 4, 4);
    let output10 = graph.conv(input1, filter, 1, 1, 1, 1);
    let output11 = graph.conv(input1, filter, 1, 1, 4, 1);

    graph.output_all(&[output00, output01, output10, output11]);

    test_all(
        &graph,
        0,
        &[
            manual_tensor::<f32, _>((1, 1, 1, 1), vec![1.0]),
            linspace_tensor((1, 1, 8, 8)),
        ],
        None,
    );
}

#[test]
fn affine_padding() {
    let input_data = linspace_tensor((8, 3, 8, 8));
    let filter_data = linspace_tensor((5, 3, 3, 3));

    let mut graph = Graph::new();

    let filter = graph.constant::<f32>(
        Shape::fixed(filter_data.shape()),
        filter_data.unwrap_f32().unwrap().to_owned().into_raw_vec(),
    );
    let bias_0 = graph.constant::<f32>(shape![1, 3, 1, 1], linspace_vec(3));
    let bias_1 = graph.constant::<f32>(shape![1, 5, 1, 1], linspace_vec(5));

    let mut curr = graph.input(Shape::fixed(&input_data.shape()), DType::F32);
    curr = graph.add(curr, bias_0);
    curr = graph.conv(curr, filter, 1, 1, 1, 1);
    curr = graph.add(curr, bias_1);
    graph.output(curr);

    test_all(&graph, 0, &[input_data], None)
}

#[test]
fn pre_act_resnet() {
    let mut graph = Graph::new();

    let input_data = linspace_tensor((8, 3, 8, 8));
    let input = graph.input(Shape::fixed(input_data.shape()), DType::F32);

    let filter_initial = graph.constant::<f32>(shape![5, 3, 3, 3], linspace_vec(5 * 3 * 3 * 3));
    let filter_tower = graph.constant::<f32>(shape![5, 5, 3, 3], linspace_vec(5 * 5 * 3 * 3));
    let filter_policy = graph.constant::<f32>(shape![2, 5, 1, 1], linspace_vec(5 * 2));

    let mut tower = graph.conv(input, filter_initial, 1, 1, 1, 1);
    for _ in 0..2 {
        let mut curr = channel_batchnorm(&mut graph, tower);
        curr = graph.clamp::<f32>(curr, 0.0, 6.0);
        curr = graph.conv(curr, filter_tower, 1, 1, 1, 1);
        curr = graph.clamp::<f32>(curr, 0.0, 6.0);
        curr = graph.conv(curr, filter_tower, 1, 1, 1, 1);
        tower = graph.add(curr, tower);
    }

    let policy = graph.conv(tower, filter_policy, 1, 1, 0, 0);

    graph.output(tower);
    graph.output(policy);

    test_all(&graph, 0, &[input_data], None)
}

fn channel_batchnorm(graph: &mut Graph, input: Value) -> Value {
    let [_, c, _, _] = graph[input].shape.unwrap_4();
    let c = c.unwrap_fixed("Dummy BN channel count");

    let const_shape = shape![1, c, 1, 1];

    let mean = graph.constant::<f32>(const_shape.clone(), linspace_vec(c));
    let var = graph.constant::<f32>(const_shape.clone(), linspace_vec(c));
    let scale = graph.constant::<f32>(const_shape.clone(), linspace_vec(c));
    let bias = graph.constant::<f32>(const_shape.clone(), linspace_vec(c));

    let mut curr = input;
    curr = graph.add(curr, mean);
    curr = graph.mul(curr, var);
    curr = graph.mul(curr, scale);
    curr = graph.add(curr, bias);
    curr
}

#[test]
fn fuse_res() {
    let mut graph = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);

    let input = graph.input(shape![10, 4, 8, 8], DType::F32);
    let other = graph.input(shape![10, 4, 8, 8], DType::F32);
    let filter = graph.constant::<f32>(shape![4, 4, 3, 3], linspace_vec(4 * 4 * 3 * 3));

    let mut curr = input;
    curr = graph.conv(curr, filter, 1, 1, 1, 1);
    curr = graph.add(curr, other);
    curr = graph.clamp::<f32>(curr, 0.0, f32::INFINITY);
    graph.output(curr);

    test_all(
        &graph,
        0,
        &[
            rng_tensor_f32((10, 4, 8, 8), &mut rng),
            rng_tensor_f32((10, 4, 8, 8), &mut rng),
        ],
        None,
    );
}

#[test]
fn concat() {
    let mut graph = Graph::new();

    let a = graph.constant::<f32>(shape![2, 3, 4], linspace_vec(2 * 3 * 4));
    let b = graph.constant::<f32>(shape![2, 1, 4], linspace_vec(2 * 1 * 4));
    let c = graph.constant::<f32>(shape![2, 8, 4], linspace_vec(2 * 8 * 4));

    let result = graph.concat(vec![a, b, c], 1, None, None);
    graph.output(result);

    test_all(&graph, 0, &[], None);
}

#[test]
fn pad() {
    let mut graph = Graph::new();

    let a = graph.constant::<f32>(shape![2, 3, 4], linspace_vec(2 * 3 * 4));

    let zero = graph.scalar(0f32);
    let b = graph.pad(a, &[(0, 0), (1, 2), (2, 0)], zero);
    graph.output(b);

    test_all(&graph, 0, &[], None);
}

// TODO permute with different data types
// TODO all operations with different data types!
#[test]
fn permute() {
    let mut graph = Graph::new();

    let a = graph.constant::<f32>(shape![2, 3, 4, 5], range_vec(2 * 3 * 4 * 5));
    for (i, permutation) in (0..4).permutations(4).enumerate() {
        println!("Output {} is permutation {:?}", i, permutation);

        let result = graph.permute(a, permutation);
        graph.output(result);
    }

    test_all(&graph, 0, &[], None);
}

#[test]
fn chain() {
    // child implements y = x * 2.0
    let mut child = Graph::new();
    {
        let child_x = child.input(shape![2], DType::F32);
        let child_w = child.constant::<f32>(shape![1], vec![2.0]);
        let child_y = child.mul(child_x, child_w);
        child.output(child_y);
    }

    // parent implements y = child(x + 3.0)
    let mut parent = Graph::new();
    let parent_x = parent.input(shape![2], DType::F32);
    let parent_w = parent.constant::<f32>(shape![1], vec![3.0]);
    let parent_z = parent.add(parent_x, parent_w);
    let parent_y = parent.call(&child, &[parent_z]);

    assert_eq!(parent_y.len(), 1);
    parent.output(parent_y[0]);

    test_all(
        &parent,
        0,
        &[manual_tensor::<f32, _>(2, vec![1.0, 2.0])],
        Some(&[manual_tensor::<f32, _>(2, vec![8.0, 10.0])]),
    )
}

#[test]
fn repeated_conv() {
    let mut graph = Graph::new();

    // weights must be different, otherwise the graph builder already deduplicates nodes
    let weight0 = graph.constant::<f32>(shape![4, 4, 3, 3], Array1::linspace(-1.0, 1.0, 4 * 4 * 3 * 3).to_vec());
    let weight1 = graph.constant::<f32>(shape![4, 4, 3, 3], Array1::linspace(-2.0, 2.0, 4 * 4 * 3 * 3).to_vec());

    let input = graph.input(shape![Size::BATCH, 4, 8, 8], DType::F32);

    let x1 = graph.conv(input, weight0, 1, 1, 1, 1);
    let x2 = graph.conv(x1, weight0, 1, 1, 1, 1);
    let x3 = graph.conv(x2, weight0, 1, 1, 1, 1);
    let x4 = graph.conv(x3, weight0, 1, 1, 1, 1);

    let y1 = graph.conv(input, weight1, 1, 1, 1, 1);
    let y2 = graph.conv(y1, weight1, 1, 1, 1, 1);
    let y3 = graph.conv(y2, weight1, 1, 1, 1, 1);
    let y4 = graph.conv(y3, weight1, 1, 1, 1, 1);

    graph.output_all(&[x4, y4]);

    test_all(&graph, 2, &[linspace_tensor((2, 4, 8, 8))], None);
}

#[test]
fn strided_conv() {
    let mut graph = Graph::new();
    let mut rng = StdRng::seed_from_u64(0);

    let input = graph.input(shape![Size::BATCH, 3, 8, 8], DType::F32);
    let filter = graph.constant::<f32>(shape![16, 3, 3, 3], rng_vec(16 * 3 * 3 * 3, &mut rng));
    let output = graph.conv(input, filter, 2, 4, 1, 1);
    assert_eq!(graph[output].shape, shape![Size::BATCH, 16, 4, 2]);

    let batch_size = 4;
    let input = rng_tensor_f32(graph[input].shape.eval(batch_size).dims.as_slice(), &mut rng);
    test_all(&graph, batch_size, &[input], None);
}

#[test]
fn softmax() {
    let mut graph = Graph::new();

    let input = graph.input(shape![3, 3], DType::F32);

    let result0 = graph.softmax(input, 0);
    let result1 = graph.softmax(input, 1);

    let scale = graph.scalar::<f32>(2.0);
    let input_scaled = graph.mul(input, scale);
    let result2 = graph.softmax(input_scaled, 0);

    graph.output_all(&[result0, result1, result2]);

    let input_data = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1.0, -1.0, f32::NEG_INFINITY];
    test_all(&graph, 0, &[manual_tensor::<f32, _>((3, 3), input_data)], None);
}

#[test]
fn reduce_easy() {
    let mut graph = Graph::new();

    let input = graph.input(shape![4, 3], DType::F32);

    for &axis in &[0, 1] {
        for &op in ReduceOp::ALL {
            let result = graph.reduce(input, vec![axis], op);
            graph.output(result);
        }
    }

    let input_data = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1.0, -1.0, -1.0 / 0.0, 0.0, 1.0, 2.0];
    test_all(&graph, 0, &[manual_tensor::<f32, _>((4, 3), input_data)], None);
}

#[test]
fn reduce_mixed() {
    let mut graph = Graph::new();

    let input = graph.input(shape![12, 3, 7, 9, 13], DType::F32);
    let mixed = graph.permute(input, vec![0, 3, 2, 1, 4]);
    let output = graph.reduce(mixed, vec![1, 2, 4], ReduceOp::Sum);
    graph.output(output);

    test_all(&graph, 0, &[linspace_tensor((12, 3, 7, 9, 13))], None);
}

#[test]
fn reduce_single() {
    let mut graph = Graph::new();

    let input = graph.input(shape![4], DType::F32);
    let output = graph.reduce(input, vec![0], ReduceOp::Sum);
    graph.output(output);

    test_all(&graph, 0, &[linspace_tensor(4)], None);
}

#[test]
fn softmax_single() {
    let mut graph = Graph::new();

    let input = graph.input(shape![4], DType::F32);
    let output = graph.softmax(input, 0);
    graph.output(output);

    test_all(&graph, 0, &[linspace_tensor(4)], None);
}

#[test]
fn layernorm_fused() {
    let input_shape = shape![Size::BATCH, 8, 32];
    let eps = 1e-5;
    let axis = 2;

    let graph = {
        let mut graph = Graph::new();

        let input = graph.input(input_shape.clone(), DType::F32);
        let reduced_shape = shape![Size::BATCH, 8, 1];

        let const_2 = graph.scalar::<f32>(2.0);
        let const_eps = graph.scalar::<f32>(eps);

        let mean = graph.reduce(input, vec![axis], ReduceOp::Mean);
        let mean = graph.view(mean, reduced_shape.clone());
        let zeroed = graph.sub(input, mean);

        let pow = graph.pow(zeroed, const_2);
        let var = graph.reduce(pow, vec![axis], ReduceOp::Mean);
        let var = graph.view(var, reduced_shape.clone());
        let var = graph.add(var, const_eps);

        let std = graph.unary(UnaryOp::Sqrt, var);
        let result = graph.binary(BinaryOp::Div, zeroed, std);

        graph.output(result);

        graph
    };

    println!("Input graph:");
    println!("{}", graph);

    {
        println!("Checking for layernorm fusion");
        // check whether we correctly fuse everything into a single layernorm operation
        let optimized = optimize_graph(&graph, Default::default());
        println!("Optimized graph:\n{}:\n\n", optimized);
        let optimized_values = optimized.values().collect_vec();
        assert_eq!(optimized_values.len(), 2);
        let input = optimized_values[0];
        let output = optimized_values[1];

        assert_eq!(optimized[input].operation, Operation::Input { index: 0 });
        assert_eq!(
            optimized[output].operation,
            Operation::Layernorm {
                input,
                axis,
                eps: Total::from(eps),
            }
        );
    }

    test_all(&graph, 2, &[linspace_tensor((2, 8, 32))], None);
}

#[test]
fn scalar_scalar() {
    let mut graph = Graph::new();

    let input = graph.input(Shape::SCALAR, DType::F32);
    let result = graph.unary(UnaryOp::Exp, input);
    graph.output(result);

    test_all(&graph, 0, &[manual_tensor::<f32, _>((), vec![2.0])], None);
}

#[test]
fn split_stride() {
    let mut graph = Graph::new();

    let len = 4;

    let x = graph.input(shape![2 * len], DType::F32);
    // let y1 = graph.slice(x, 0, SliceRange::new(0, 2 * len, 2));
    // graph.output(y1);
    let y2 = graph.slice(x, 0, SliceRange::new(1, 2 * len + 1, 2));
    graph.output(y2);

    let mut rng = StdRng::seed_from_u64(0);
    let input = rng_tensor_f32(2 * len, &mut rng);
    println!("{:?}", input);
    test_all(&graph, 0, &[input], None);
}

// TODO debug failing scalar fusion? did this never fuse or did we break something?
#[test]
fn complex_multiply_stride() {
    complex_multiply(false);
}

#[test]
fn complex_multiply_new_axis() {
    complex_multiply(true);
}

fn complex_multiply(new_axis: bool) {
    let batch = 8;
    let len = 32;
    let axis = 1;

    let mut graph = Graph::new();
    let a = graph.input(shape![batch, len * 2], DType::F32);
    let b = graph.input(shape![batch, len * 2], DType::F32);

    // TODO create utility function for this entire operation, "complex_multiply"
    // TODO this is a good test for scalar fusion, especially the final concat
    let result = {
        let (ar, ai, br, bi);

        if new_axis {
            println!("Using new axis slice");

            let a_split = graph.view(a, shape![batch, len, 2]);
            let b_split = graph.view(b, shape![batch, len, 2]);

            let ar_split = graph.slice(a_split, axis + 1, SliceRange::single(0));
            let ai_split = graph.slice(a_split, axis + 1, SliceRange::single(1));
            let br_split = graph.slice(b_split, axis + 1, SliceRange::single(0));
            let bi_split = graph.slice(b_split, axis + 1, SliceRange::single(1));

            ar = graph.view(ar_split, shape![batch, len]);
            ai = graph.view(ai_split, shape![batch, len]);
            br = graph.view(br_split, shape![batch, len]);
            bi = graph.view(bi_split, shape![batch, len]);
        } else {
            println!("Using stride 2 slice");

            ar = graph.slice(a, axis, SliceRange::new(0, len * 2, 2));
            ai = graph.slice(a, axis, SliceRange::new(1, len * 2 + 1, 2));
            br = graph.slice(b, axis, SliceRange::new(0, len * 2, 2));
            bi = graph.slice(b, axis, SliceRange::new(1, len * 2 + 1, 2));
        }

        let ar_br = graph.binary(BinaryOp::Mul, ar, br);
        let ai_bi = graph.binary(BinaryOp::Mul, ai, bi);
        let ar_bi = graph.binary(BinaryOp::Mul, ar, bi);
        let ai_br = graph.binary(BinaryOp::Mul, ai, br);

        let r = graph.binary(BinaryOp::Sub, ar_br, ai_bi);
        let i = graph.binary(BinaryOp::Add, ar_bi, ai_br);

        // interleave again
        // TODO create a graph utility function for this, "interleave"
        let concat_values = vec![
            graph.view(r, shape![batch, len, 1]),
            graph.view(i, shape![batch, len, 1]),
        ];

        let result_extra = graph.concat(concat_values, axis + 1, None, None);
        let result = graph.view(result_extra, shape![batch, len * 2]);
        result
    };

    graph.output(result);

    let mut rng = StdRng::seed_from_u64(0);
    let input_a = Array::from_shape_simple_fn((batch, len * 2), || rng.gen());
    let input_b = Array::from_shape_simple_fn((batch, len * 2), || rng.gen());
    let output = Array::from_shape_fn((batch, len * 2), |(bi, ni_split)| {
        let ni = ni_split / 2;
        let get_r = ni_split % 2 == 0;

        let ar = input_a[(bi, 2 * ni)];
        let ai = input_a[(bi, 2 * ni + 1)];
        let br = input_b[(bi, 2 * ni)];
        let bi = input_b[(bi, 2 * ni + 1)];

        if get_r {
            ar * br - ai * bi
        } else {
            ar * bi + ai * br
        }
    });

    if batch * len < 16 {
        println!("input_a: {:?}", input_a.iter().collect_vec());
        println!("input_b: {:?}", input_b.iter().collect_vec());
        println!("output: {:?}", output.iter().collect_vec());
    }

    test_all(
        &graph,
        0,
        &[
            DTensor::F32(input_a.into_shared().into_dyn()),
            DTensor::F32(input_b.into_shared().into_dyn()),
        ],
        Some(&[DTensor::F32(output.into_shared().into_dyn())]),
    );
}
