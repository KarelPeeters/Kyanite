use std::cmp::max;
use std::time::Instant;

use clap::Parser;
use itertools::{Itertools, izip};

use cuda_nn_eval::Device;
use cuda_nn_eval::executor::CudnnExecutor;
use nn_graph::graph::Graph;
use nn_graph::onnx::load_graph_from_onnx_path;
use nn_graph::optimizer::{optimize_graph, OptimizerSettings};

#[derive(Debug, clap::Parser)]
struct Args {
    #[clap(short, long)]
    batch_size: i32,
    #[clap(short, long)]
    optimize: bool,
    #[clap(short, long)]
    graph: bool,
    #[clap(short, long)]
    print: bool,

    path: String,
}

const TEST_BATCH_SIZES: &[usize] = &[1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
const TEST_BATCH_ITERATIONS: &[usize] = &[100, 100, 100, 100, 100, 100, 100, 100, 50, 25, 12, 6, 3];

const ITERATIONS: usize = 100;

fn main() {
    let Args { batch_size, optimize, graph: use_graph, print, path } = Args::parse();

    if cfg!(debug_assertions) {
        println!("Warning: debug assertions are enabled, maybe this binary is not optimized either?");
    }
    if !use_graph {
        println!("Warning: not using cuda graph mode");
    }

    let abs_path = std::fs::canonicalize(path).unwrap();
    println!("Loading graph '{:?}'", abs_path);
    let loaded_graph = load_graph_from_onnx_path(abs_path);

    let graph = if optimize {
        println!("Optimizing graph");
        optimize_graph(&loaded_graph, OptimizerSettings::default())
    } else {
        println!("Warning: not optimizing graph");
        loaded_graph
    };

    if print {
        println!("{}", graph);
    }

    if batch_size < 1 {
        profile_different_batch_sizes(&graph, use_graph);
    } else {
        profile_single_batch_size(&graph, use_graph, batch_size as usize);
    }
}

fn profile_different_batch_sizes(graph: &Graph, use_graph: bool) {
    let mut result = vec![];

    for (&batch_size, &iterations) in izip!(TEST_BATCH_SIZES, TEST_BATCH_ITERATIONS) {
        println!("Testing batch size {} with {} iterations", batch_size, iterations);

        let mut executor = CudnnExecutor::new(Device::new(0), &graph, batch_size, use_graph);
        let inputs = dummy_inputs(&graph, batch_size);
        let inputs = inputs.iter().map(|v| &**v).collect_vec();

        for _ in 0..max(1, iterations / 10) {
            executor.evaluate(&inputs);
        }

        let start = Instant::now();
        for _ in 0..iterations {
            executor.evaluate(&inputs);
        }
        let delta = (Instant::now() - start).as_secs_f32();
        let throughput = (batch_size * iterations) as f32 / delta;

        println!("  throughput: {} evals/s", throughput);
        result.push((batch_size, throughput));
    }

    println!("{:?}", result);
}

fn profile_single_batch_size(graph: &Graph, use_graph: bool, batch_size: usize) {
    let mut executor = CudnnExecutor::new(Device::new(0), &graph, batch_size, use_graph);

    let inputs = dummy_inputs(&graph, batch_size);
    let inputs = inputs.iter().map(|v| &**v).collect_vec();

    println!("Warmup");
    for _ in 0..max(1, ITERATIONS / 10) {
        executor.evaluate(&inputs);
    }

    println!("Throughput test");
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        executor.evaluate(&inputs);
    }
    let delta = (Instant::now() - start).as_secs_f32();
    let throughput = (batch_size * ITERATIONS) as f32 / delta;

    println!("Profiling");
    executor.set_profile(true);
    executor.evaluate(&inputs);
    println!("{}", executor.last_profile().unwrap());

    // only print this now so it's easily visible
    println!("Throughput: {} evals/s", throughput);
}

fn dummy_inputs(graph: &Graph, batch_size: usize) -> Vec<Vec<f32>> {
    graph.inputs().iter().map(|&v| {
        vec![0f32; graph[v].shape.size().eval(batch_size)]
    }).collect_vec()
}
