use crate::root::runner::{test_elementwise, test_elementwise_pair};
use nn_graph::graph::ElementOp;

#[test]
fn all() {
    for &op in ElementOp::ALL {
        test_elementwise_pair(|a, b| op.map(a, b), |g, a, b| g.ele(op, a, b))
    }
}

#[test]
fn clamp() {
    for min in [f32::NEG_INFINITY, 0.0] {
        for max in [f32::INFINITY, 6.0] {
            println!("Testing clamp({}, {})", min, max);
            test_elementwise(|a| a.clamp(min, max), |g, a| g.clamp(a, min, max))
        }
    }
}
