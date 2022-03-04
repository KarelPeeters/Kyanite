use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use cuda_nn_eval::executor::CudnnExecutor;
use cuda_nn_eval::Device;
use nn_graph::graph::Graph;
use nn_graph::shape::Shape;

fn bench_copy(c: &mut Criterion) {
    let sizes = [
        // number of bytes for a batch of 128 chess boards
        ("chess_128", 128 * 21 * 8 * 8 * 4, 128 * (72 - 21) * 8 * 8 * 1),
    ];

    for (name, input_size, extra_output_size) in sizes {
        println!(
            "Name: {}, input_size: {}, extra_output_size: {}",
            name, input_size, extra_output_size
        );

        let mut graph = Graph::new();
        let input = graph.input(Shape::fixed(&[input_size]));
        let constant = graph.constant(Shape::fixed(&[extra_output_size]), vec![0.5; extra_output_size]);
        graph.output_all(&[input, constant]);
        println!("{}", graph);

        let device = Device::new(0);
        let mut executor = CudnnExecutor::new(device, &graph, 1, false);

        println!("{:?}", executor);

        let input = vec![1.0; input_size];

        c.bench_function(&format!("copy {}", name), |b| {
            b.iter(|| {
                black_box(executor.evaluate(&[&input]).len());
            })
        });
    }
}

criterion_group!(
    name=benches;
    config=Criterion::default().measurement_time(Duration::from_secs(10));
    targets=bench_copy
);
criterion_main!(benches);
