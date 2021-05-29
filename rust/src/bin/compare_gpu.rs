use sttt_zero::network::Network;
use tch::Device;
use sttt::board::Board;
use std::time::Instant;
use std::cmp::min;

fn main() {
    let mut cpu_network = Network::load("../data/esat/trained_model_10_epochs.pt", Device::Cpu);
    let mut cuda_network = Network::load("../data/esat/trained_model_10_epochs.pt", Device::Cuda(0));

    let batch_sizes = [1, 2, 4, 10, 50, 100, 1000, 3000, 5000];

    println!("CPU vs GPU throughput experiment");
    for &batch_size in &batch_sizes {
        println!("Batch size: {}", batch_size);
        let rounds = min(500, 10_000 / batch_size);

        let batch = vec![Board::new(); batch_size];
        println!("CPU: {}", (batch_size as f32) / avg_time(rounds, &mut || cpu_network.evaluate_all(&batch)));
        println!("GPU: {}", (batch_size as f32) / avg_time(rounds, &mut || cuda_network.evaluate_all(&batch)));
    }
}

fn avg_time<R>(rounds: usize, f: &mut impl FnMut() -> R) -> f32 {
    let start = Instant::now();
    for _ in 0..rounds {
        f();
    }
    (Instant::now() - start).as_secs_f32() / (rounds as f32)
}