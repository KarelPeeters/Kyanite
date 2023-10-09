extern crate prost_build;

use prost_build::Config;
use std::io::ErrorKind;

fn main() -> std::io::Result<()> {
    let path_curr = std::env::current_dir()?;
    let path_graph = path_curr.join("kn-graph");

    let path_dir = path_graph.join("proto");
    let path_proto = path_dir.join("onnx.proto3");
    assert!(path_proto.exists(), "{:?} does not exist", path_proto);

    let path_out = path_graph.join("src/onnx");

    Config::new()
        .out_dir(&path_out)
        .compile_protos(&[path_proto], &[path_dir])?;

    // TODO figure out a way to immediately set the output filename instead
    // delete file if it exists
    match std::fs::remove_file(path_out.join("proto.rs")) {
        Ok(_) => {}
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    std::fs::rename(path_out.join("onnx.rs"), path_out.join("proto.rs"))?;

    Ok(())
}
