use crate::root::runner::test_onnx_bin;

#[test]
fn simple() {
    test_onnx_bin(
        include_bytes!("../data/simple-sttt-1x64.onnx"),
        include_bytes!("../data/simple-sttt-1x64.bin"),
    )
}

#[test]
fn simple_bn() {
    test_onnx_bin(
        include_bytes!("../data/simple-bn-sttt-1x64.onnx"),
        include_bytes!("../data/simple-bn-sttt-1x64.bin"),
    )
}

#[test]
fn pre() {
    test_onnx_bin(
        include_bytes!("../data/pre-sttt-4x8.onnx"),
        include_bytes!("../data/pre-sttt-4x8.bin"),
    )
}
