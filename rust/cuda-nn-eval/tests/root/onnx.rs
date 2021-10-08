use align_data::include_aligned;

use crate::root::runner::{OnnxPair, test_onnx_pair};

static SIMPLE: OnnxPair = OnnxPair {
    onnx: include_bytes!("../data/simple-sttt-1x64.onnx"),
    bin: include_aligned!(f32, "../data/simple-sttt-1x64.bin"),
};

static PRE: OnnxPair = OnnxPair {
    onnx: include_bytes!("../data/pre-sttt-4x8.onnx"),
    bin: include_aligned!(f32, "../data/pre-sttt-4x8.bin"),
};

#[test]
fn simple() {
    test_onnx_pair(&SIMPLE)
}

#[test]
fn pre() {
    test_onnx_pair(&PRE)
}