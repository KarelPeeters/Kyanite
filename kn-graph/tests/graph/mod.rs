use kn_graph::graph::Graph;
use kn_graph::shape;

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