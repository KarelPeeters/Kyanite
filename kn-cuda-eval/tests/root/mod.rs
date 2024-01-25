use std::ffi::OsStr;

use kn_graph::onnx::load_graph_from_onnx_path;

use crate::root::runner::test_all;

mod runner;
mod tensor_utils;

mod elementwise;
mod opt;

mod graphs;
mod onnx;

mod slow;

#[test]
fn onnx_tests() {
    let path_folder = r#"C:\Documents\Programming\STTT\Kyanite\kn-python\models"#;

    for entry in std::fs::read_dir(path_folder).unwrap() {
        let entry = entry.unwrap();
        let path_onnx = entry.path();
        if path_onnx.extension() != Some(OsStr::new("onnx")) {
            continue;
        }

        println!("Testing {:?}", path_onnx);

        let graph = load_graph_from_onnx_path(path_onnx, false).unwrap();
        let inputs = graph.dummy_zero_inputs(0);

        test_all(&graph, 0, &inputs, None);
    }
}