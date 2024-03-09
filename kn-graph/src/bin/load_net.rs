use std::process::exit;

use itertools::Itertools;

use kn_graph::dot::graph_to_svg;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::{optimize_graph, OptimizerSettings};

fn main() {
    // parse args
    let args = std::env::args().collect_vec();

    let optimize = args.iter().any(|a| a == "-o");
    let args = args.iter().filter(|&a| a != "-o").collect_vec();

    if args.len() != 2 && args.len() != 3 {
        eprintln!("Usage: load_net [-o] <path.onnx> [path.svg]");
        exit(1);
    }
    let path = &args[1];
    let out_path = args.get(2);

    // load graph
    let mut graph = load_graph_from_onnx_path(path, true).unwrap();

    // optimize graph
    if optimize {
        graph = optimize_graph(&graph, OptimizerSettings::default());
    }

    // outputs
    println!("{}", graph);

    if let Some(out_path) = out_path {
        graph_to_svg(out_path, &graph, false, true).unwrap();
    }
}