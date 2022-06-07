use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Barrier;

use clap::Parser;
use itertools::{enumerate, Itertools};
use rand::distributions::Standard;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use cuda_nn_eval::executor::CudaExecutor;
use cuda_nn_eval::tester::{assert_tensors_match, check_tensors_match};
use cuda_sys::wrapper::handle::Device;
use nn_graph::cpu::Tensor;
use nn_graph::graph::Graph;
use nn_graph::onnx::load_graph_from_onnx_path;
use nn_graph::optimizer::{optimize_graph, OptimizerSettings};

#[derive(Debug, clap::Parser)]
struct Args {
    #[clap(short, long)]
    batch_size: usize,
    #[clap(short, long)]
    device: Vec<i32>,
    #[clap(short, long)]
    threads: usize,

    path: String,
}

type IOPair = (Vec<Tensor>, Vec<Tensor>);

fn main() {
    let Args {
        batch_size,
        device: devices,
        threads,
        path,
    } = Args::parse();

    let devices = if devices.is_empty() {
        Device::all().collect_vec()
    } else {
        devices.into_iter().map(Device::new).collect_vec()
    };

    println!("Using devices {:?}", devices);
    assert!(!devices.is_empty());

    let graph = load_graph_from_onnx_path(path);
    let graph = optimize_graph(&graph, OptimizerSettings::default());

    println!("Generating io pairs");
    let mut rng = SmallRng::from_entropy();
    let pairs = generate_io_pairs(*devices.first().unwrap(), &graph, batch_size, 16, &mut rng);

    println!("Launching threads");
    let barrier = Barrier::new(devices.len() * threads);
    let failed = AtomicBool::new(false);

    crossbeam::scope(|s| {
        let graph = &graph;
        let pairs = &pairs;
        let barrier = &barrier;
        let failed = &failed;

        for (di, device) in enumerate(devices) {
            for thread in 0..threads {
                s.builder()
                    .name(format!("thread-{}-{}", di, thread))
                    .spawn(move |_| {
                        device_thread_main(device, &graph, batch_size, pairs, barrier, failed);
                    })
                    .unwrap();
            }
        }
    })
    .unwrap();
}

fn generate_io_pairs(
    device: Device,
    graph: &Graph,
    batch_size: usize,
    count: usize,
    rng: &mut impl Rng,
) -> Vec<IOPair> {
    let mut executor = CudaExecutor::new(device, graph, batch_size);

    (0..count)
        .map(|_| {
            let inputs = graph
                .inputs()
                .iter()
                .map(|&v| {
                    let shape = graph[v].shape.eval(batch_size);
                    let data = rng.sample_iter(Standard).take(shape.size()).collect_vec();
                    Tensor::from_shape_vec(shape.dims, data).unwrap()
                })
                .collect_vec();

            let outputs = executor.evaluate_tensors(&inputs);

            (inputs, outputs)
        })
        .collect_vec()
}

fn device_thread_main(
    device: Device,
    graph: &Graph,
    batch_size: usize,
    pairs: &[IOPair],
    barrier: &Barrier,
    failed: &AtomicBool,
) {
    let rng = &mut SmallRng::from_entropy();
    let thread_name = std::thread::current().name().unwrap().to_owned();

    println!("{}: Building executor", thread_name);
    let mut executor = CudaExecutor::new(device, graph, batch_size);

    // wait for all executors to be constructed, since the memcopies
    // they cause cannot run in parallel with other threads
    barrier.wait();

    for i in 0.. {
        if failed.load(Ordering::SeqCst) {
            break;
        }

        let (input, expected) = pairs.choose(rng).unwrap();
        let actual = executor.evaluate_tensors(&input);

        if check_tensors_match(&expected, &actual).is_err() {
            failed.store(true, Ordering::SeqCst);

            eprintln!("{} Iteration {} failed", thread_name, i);
            assert_tensors_match(expected, &actual, false);
        } else {
            println!("{}: Iteration {} success", thread_name, i);
        }
    }
}
