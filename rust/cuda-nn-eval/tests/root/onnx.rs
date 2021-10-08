use crate::root::runner::{OnnxPair, test_onnx_pair};

const SIMPLE: OnnxPair = OnnxPair {
    onnx: include_bytes!("../data/simple-sttt-1x64.onnx"),
    bin: include_bytes!("../data/simple-sttt-1x64.bin"),
};

#[test]
fn simple() {
    test_onnx_pair(&SIMPLE)
}
