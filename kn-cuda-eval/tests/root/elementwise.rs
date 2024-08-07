use kn_graph::dtype::{DTensor, IntoDScalar};
use kn_graph::graph::{BinaryOp, Graph, UnaryOp};
use kn_graph::shape;
use kn_graph::shape::{Shape, Size};

use crate::root::runner::{test_all, test_elementwise, test_elementwise_pair};
use crate::root::tensor_utils::manual_tensor;

#[test]
fn unary() {
    // TODO build single graph for all cases
    // TODO other dtypes
    for &op in UnaryOp::ALL {
        test_elementwise(
            |x| f32::from_dscalar(op.map(x.to_dscalar())).unwrap(),
            |g, a| g.unary(op, a),
        );
    }
}

#[test]
fn unary_fixed() {
    let graph = {
        let mut graph = Graph::new();
        let x = graph.input(shape![Size::BATCH], f32::DTYPE);
        let y = graph.unary(UnaryOp::Softplus, x);
        graph.output(y);
        graph
    };

    let x: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let y: Vec<f32> = vec![0.126928, 0.313262, 0.693147, 1.313262, 2.126928];

    let x = manual_tensor(x.len(), x);
    let y = manual_tensor(y.len(), y);
    test_all(&graph, x.len(), &[x], Some(&[y]));
}

#[test]
fn binary() {
    // TODO build single graph for all cases
    // TODO other dtypes
    for &op in BinaryOp::ALL {
        test_elementwise_pair(|a, b| op.map_t(a, b), |g, a, b| g.binary(op, a, b))
    }
}

#[test]
fn clamp() {
    // TODO build single graph for all cases
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
    let mut state = CastState::new();

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

    state.run();
}

#[test]
fn bit_cast() {
    let mut state = CastState::new();

    // identity
    state.bit(5u8, 5u8);
    state.bit(5u32, 5u32);
    state.bit(5f32, 5f32);
    state.bit(0xdeadbeefdeadbeefu64, 0xdeadbeefdeadbeefu64);

    // float <-> int
    state.bit(1f32, 1f32.to_bits());
    state.bit(-1f32, (-1f32).to_bits());

    state.run();
}

// TODO test tricky double casts, eg. `i8->u16->u32` vs `i8->i16->u32`

// TODO turn this into more general elementwise ops test builder, use for other unary/binary tests too
struct CastState {
    graph: Graph,
    inputs: Vec<DTensor>,
    outputs: Vec<DTensor>,
    ran: bool,
}

impl CastState {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            inputs: vec![],
            outputs: vec![],
            ran: false,
        }
    }

    fn value<X: IntoDScalar, Y: IntoDScalar>(&mut self, x: X, y: Y) {
        let xv = self.graph.input(Shape::SCALAR, X::DTYPE);
        let yv = self.graph.unary(UnaryOp::ValueCast(Y::DTYPE), xv);
        self.graph.output(yv);

        self.inputs.push(x.to_dscalar().to_tensor());
        self.outputs.push(y.to_dscalar().to_tensor());
    }

    fn bit<X: IntoDScalar, Y: IntoDScalar>(&mut self, x: X, y: Y) {
        let xv = self.graph.input(Shape::SCALAR, X::DTYPE);
        let yv = self.graph.unary(UnaryOp::BitCast(Y::DTYPE), xv);
        self.graph.output(yv);

        self.inputs.push(x.to_dscalar().to_tensor());
        self.outputs.push(y.to_dscalar().to_tensor());
    }

    fn run(&mut self) {
        self.ran = true;
        test_all(&self.graph, 0, &self.inputs, Some(&self.outputs));
    }
}

impl Drop for CastState {
    fn drop(&mut self) {
        if !self.ran {
            panic!("CastState didn't run");
        }
    }
}
