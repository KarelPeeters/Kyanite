use std::cmp::max;
use std::time::Instant;

use clap::Parser;
use itertools::{Itertools, izip};

use kn_cuda_eval::CudaDevice;
use kn_cuda_eval::executor::CudaExecutor;
use kn_graph::cpu::cpu_eval_graph;
use kn_graph::graph::Graph;
use kn_graph::onnx::load_graph_from_onnx_path;
use kn_graph::optimizer::{optimize_graph, OptimizerSettings};
use kn_graph::shape::Size;

#[derive(Debug, clap::Parser)]
struct Args {
    #[clap(short, long, default_value_t = 0)]
    batch_size: i32,
    #[clap(short, long)]
    optimize: bool,
    #[clap(short, long)]
    print: bool,
    #[clap(short, long)]
    cpu: bool,

    #[clap(short, long)]
    device: Option<i32>,
    #[clap(short, long)]
    skip_io: bool,

    #[clap(long, short)]
    n: Option<usize>,

    path: String,
}

const TEST_BATCH_SIZES: &[usize] = &[1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
const TEST_BATCH_ITERATIONS: &[usize] = &[100, 100, 100, 100, 100, 100, 100, 100, 50, 25, 12, 8, 4];

const DEFAULT_ITERATIONS: usize = 100;

fn main() {
    let Args {
        batch_size,
        optimize,
        print,
        n,
        path,
        cpu,
        device,
        skip_io,
    } = Args::parse();

    let n = n.unwrap_or(DEFAULT_ITERATIONS);
    let device = CudaDevice::new(device.unwrap_or(0)).unwrap();

    let mut warnings = vec![];
    let mut warn = |s: &str| {
        warnings.push(s.to_owned());
        eprintln!("{}", s);
    };

    if cfg!(debug_assertions) {
        warn("Warning: debug assertions are enabled, maybe this binary is not optimized either?");
    }

    let abs_path = std::fs::canonicalize(path).unwrap();
    println!("Loading graph '{:?}'", abs_path);
    let loaded_graph = load_graph_from_onnx_path(abs_path, true).unwrap();

    let any_input_no_batch = loaded_graph.inputs().iter().any(|&input| {
        let batch_count = loaded_graph[input].shape.dims.iter().filter(|&&d| d == Size::BATCH).count();
        batch_count != 1
    });
    if any_input_no_batch {
        let input_shapes = loaded_graph.inputs().iter().map(|&input| &loaded_graph[input].shape).collect_vec();
        warn("Warning: graph has inputs without exactly one batch dimension. This messes up the evals/s stats.");
        warn(&format!("    input shapes: {:?}", input_shapes));
    }

    let graph = if optimize {
        println!("Optimizing graph");
        optimize_graph(&loaded_graph, OptimizerSettings::default())
    } else {
        warn("Warning: not optimizing graph");
        loaded_graph
    };

    if print {
        println!("{}", graph);

        if !cpu {
            assert!(batch_size > 0, "Error: must set batch size");

            let executor = CudaExecutor::new(device, &graph, batch_size as usize);
            println!("{:?}", executor);
        }
    } else {
        if batch_size < 1 {
            if cpu {
                eprintln!("Error: profiling different batch sizes for CPU not yet implemented");
            } else {
                profile_different_batch_sizes(device, &graph);
            }
        } else {
            if cpu {
                profile_single_batch_size_cpu(&graph, batch_size as usize, n)
            } else {
                profile_single_batch_size_cudnn(device, &graph, batch_size as usize, n, skip_io);
            }
        }
    }

    for warning in warnings {
        eprintln!("{}", warning);
    }
}

fn profile_different_batch_sizes(device: CudaDevice, graph: &Graph) {
    let mut result = vec![];

    for (&batch_size, &iterations) in izip!(TEST_BATCH_SIZES, TEST_BATCH_ITERATIONS) {
        println!("Testing batch size {} with {} iterations", batch_size, iterations);

        let mut executor = CudaExecutor::new(device, &graph, batch_size);
        let inputs = graph.dummy_zero_inputs(batch_size);

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

fn profile_single_batch_size_cudnn(device: CudaDevice, graph: &Graph, batch_size: usize, n: usize, skip_io: bool) {
    let mut executor = CudaExecutor::new(device, &graph, batch_size);
    let inputs = graph.dummy_zero_inputs(batch_size);

    println!("Warmup");
    for _ in 0..max(1, n / 10) {
        executor.evaluate(&inputs);
    }

    println!("Throughput test");
    let start = Instant::now();
    for _ in 0..n {
        if skip_io {
            unsafe {
                executor.run_async();
            }
        } else {
            executor.evaluate(&inputs);
        }
    }
    if skip_io {
        executor.stream().synchronize();
    }

    let delta = (Instant::now() - start).as_secs_f32();
    let throughput = (batch_size * n) as f32 / delta;

    println!("Profiling");
    executor.set_profile(true);
    executor.evaluate(&inputs);
    println!("{}", executor.last_profile().unwrap());

    // only print this now so it's easily visible
    println!("Throughput: {} evals/s", throughput);
}

fn profile_single_batch_size_cpu(graph: &Graph, batch_size: usize, n: usize) {
    let inputs = graph.dummy_zero_inputs(batch_size);

    println!("Warmup");
    for _ in 0..max(1, n / 10) {
        cpu_eval_graph(graph, batch_size, &inputs);
    }

    println!("Throughput test");
    let start = Instant::now();
    for _ in 0..n {
        cpu_eval_graph(graph, batch_size, &inputs);
    }
    let delta = (Instant::now() - start).as_secs_f32();
    let throughput = (batch_size * n) as f32 / delta;
    println!("Throughput: {} evals/s", throughput);
}
