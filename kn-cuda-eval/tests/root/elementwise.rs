use kn_graph::graph::{BinaryOp, UnaryOp};

use crate::root::runner::{test_elementwise, test_elementwise_pair};

#[test]
fn unary() {
    for &op in UnaryOp::ALL {
        test_elementwise(|x| op.map(x), |g, a| g.unary(op, a));
    }
}

#[test]
fn binary() {
    for &op in BinaryOp::ALL {
        test_elementwise_pair(|a, b| op.map(a, b), |g, a, b| g.binary(op, a, b))
    }
}

#[test]
fn clamp() {
    for min in [f32::NEG_INFINITY, 0.0] {
        for max in [f32::INFINITY, 6.0] {
            println!("Testing clamp({}, {})", min, max);
            test_elementwise(|a| a.clamp(min, max), |g, a| g.clamp::<f32>(a, min, max))
        }
    }
}
