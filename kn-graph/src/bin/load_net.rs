use std::process::exit;

use itertools::Itertools;

use kn_graph::onnx::load_graph_from_onnx_path;

fn main() {
    let args = std::env::args().collect_vec();
    if args.len() != 2 {
        eprintln!("Usage: load_net <path.onnx>");
        exit(1);
    }

    let path = &args[1];
    let graph = load_graph_from_onnx_path(path, true).unwrap();

    println!("{}", graph);
}