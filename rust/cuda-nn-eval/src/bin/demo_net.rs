use std::time::Instant;

use cuda_sys::wrapper::handle::{CudnnHandle, cuda_device_count};
use nn_cuda_eval::net::{NetDefinition, NetEvaluator};

fn main() {
    println!("Cuda device count: {}", cuda_device_count());

    let def = NetDefinition {
        tower_depth: 8,
        channels: 32,
    };

    let handle = CudnnHandle::new(0);
    let batch_size = 5000;
    let mut eval = NetEvaluator::new(handle, def, batch_size);

    let mut data = vec![0.0; batch_size as usize * def.channels as usize * 7 * 7];

    let start = Instant::now();
    let mut prev_print = Instant::now();

    for i in 0.. {
        eval.eval(&mut data);

        let now = Instant::now();
        if (now - prev_print).as_secs_f32() >= 1.0 {
            let throughput = (batch_size * i) as f32 / (now - start).as_secs_f32();
            prev_print = now;

            println!("Throughput: {} boards/s", throughput);
        }
    }
}
