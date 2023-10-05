use kn_graph::dtype::{DScalar, IntoDScalar};
use kn_graph::graph::{BinaryOp, Graph, UnaryOp};
use kn_graph::shape::Shape;

use crate::root::runner::{test_elementwise, test_elementwise_pair};

#[test]
fn unary() {
    for &op in UnaryOp::ALL {
        test_elementwise(|x| op.map_t(x), |g, a| g.unary(op, a));
    }
}

#[test]
fn binary() {
    for &op in BinaryOp::ALL {
        test_elementwise_pair(|a, b| op.map_t(a, b), |g, a, b| g.binary(op, a, b))
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

// TODO test proper tensors too? or just assume unary ops work right?
#[test]
fn value_cast() {
    let mut state = CastState { graph: Graph::new(), cases: vec![] };

    // identity
    state.value(5u8, 5u8);
    state.value(5u32, 5u32);
    state.value(5f32, 5f32);
    state.value(0xdeadbeefdeadbeefu64, 0xdeadbeefdeadbeefu64);

    // int extend
    state.value(1u8, 1u32);
    state.value(-1i8, u32::MAX);

    // int truncate
    state.value(1u32, 1u8);
    state.value(-1i32, u8::MAX);

    // float <-> int
    state.value(1f32, 1u32);
    state.value(-1f32, -1i32);
    state.value(-1f32, -1i64);
    state.value(-1f32, -1i8);
}

#[test]
fn bit_cast() {
    let mut state = CastState { graph: Graph::new(), cases: vec![] };

    // identity
    state.bit(5u8, 5u8);
    state.bit(5u32, 5u32);
    state.bit(5f32, 5f32);
    state.bit(0xdeadbeefdeadbeefu64, 0xdeadbeefdeadbeefu64);

    // float <-> int
    state.bit(1f32, 1f32.to_bits());
    state.bit(-1f32, (-1f32).to_bits());
}

struct CastState {
    graph: Graph,
    cases: Vec<(DScalar, DScalar)>,
}

impl CastState {
    fn value<X: IntoDScalar, Y: IntoDScalar>(&mut self, x: X, y: Y) {
        let xv = self.graph.input(Shape::SCALAR, X::DTYPE);
        let yv = self.graph.unary(UnaryOp::BitCast(Y::DTYPE), xv);
        self.graph.output(yv);

        self.cases.push((x.to_dscalar(), y.to_dscalar()));
    }

    fn bit<X: IntoDScalar, Y: IntoDScalar>(&mut self, x: X, y: Y) {
        let xv = self.graph.input(Shape::SCALAR, X::DTYPE);
        let yv = self.graph.unary(UnaryOp::ValueCast(Y::DTYPE), xv);
        self.graph.output(yv);

        self.cases.push((x.to_dscalar(), y.to_dscalar()));
    }
}