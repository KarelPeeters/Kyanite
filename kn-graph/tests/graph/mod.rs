use itertools::Itertools;

use kn_graph::graph::Graph;
use kn_graph::shape;
use kn_graph::shape::{ConcreteShape, Shape, Size};
use kn_graph::shape::infer_batch_size;
use kn_graph::shape::ShapeMismatch;

#[test]
fn dedup_const() {
    let mut graph = Graph::new();

    let x0 = graph.constant(shape![2], vec![1.0, 2.0]);
    let x1 = graph.constant(shape![2], vec![1.0, 2.0]);
    let x2 = graph.constant(shape![2], vec![1.0, 3.0]);
    assert_eq!(x0, x1);
    assert_ne!(x0, x2);

    let y0 = graph.constant(shape![2], vec![1.0, f32::NAN]);
    let y1 = graph.constant(shape![2], vec![1.0, f32::NAN]);
    assert_eq!(y0, y1);

    assert_ne!(x0, y0);
}

fn wrap_infer_batch_size(shapes: &[Shape], concrete: &[Vec<usize>]) -> Result<Option<usize>, ShapeMismatch> {
    infer_batch_size(
        shapes,
        &concrete.iter().map(|v| ConcreteShape::new(v.clone())).collect_vec(),
    )
}

#[test]
fn test_infer_batch_size() {
    // no batch axis
    assert_eq!(wrap_infer_batch_size(&[shape![2, 3, 4]], &[vec![2, 3, 4]]), Ok(None));
    assert_eq!(
        wrap_infer_batch_size(&[shape![2, 3, 4], shape![5, 6]], &[vec![2, 3, 4], vec![1, 6]]),
        Err(ShapeMismatch::ConstantMismatch)
    );

    // matching batch axis
    assert_eq!(
        wrap_infer_batch_size(&[shape![2, Size::BATCH]], &[vec![2, 8]]),
        Ok(Some(8))
    );
    assert_eq!(
        wrap_infer_batch_size(&[shape![Size::BATCH, Size::BATCH]], &[vec![4, 4]]),
        Ok(Some(4))
    );

    // batch axis mismatch
    assert_eq!(
        wrap_infer_batch_size(&[shape![Size::BATCH, Size::BATCH]], &[vec![4, 8]]),
        Err(ShapeMismatch::BatchConflict)
    );
    assert_eq!(
        wrap_infer_batch_size(&[shape![Size::BATCH], shape![Size::BATCH]], &[vec![4], vec![8]]),
        Err(ShapeMismatch::BatchConflict)
    );

    // matching batch exp
    assert_eq!(
        wrap_infer_batch_size(
            &[shape![
                Size::BATCH,
                Size::BATCH * Size::BATCH,
                Size::BATCH * 4,
                Size::BATCH * 8,
            ]],
            &[vec![10, 10 * 10, 10 * 4, 10 * 8]],
        ),
        Ok(Some(10))
    );

    // mismatching batch exp
    assert_eq!(
        wrap_infer_batch_size(
            &[shape![
                Size::BATCH,
                Size::BATCH * Size::BATCH,
                Size::BATCH * 4,
                Size::BATCH * 3
            ]],
            &[vec![10, 10 * 10, 10 * 4, 10 * 8]],
        ),
        Err(ShapeMismatch::BatchConflict)
    );

    // impossible batch exp
    assert_eq!(
        wrap_infer_batch_size(&[shape![Size::BATCH * 2]], &[vec![7]]),
        Err(ShapeMismatch::ImpossibleBatchValue)
    );
    assert_eq!(
        wrap_infer_batch_size(&[shape![Size::BATCH * Size::BATCH]], &[vec![15]]),
        Err(ShapeMismatch::ImpossibleBatchValue)
    );
}
