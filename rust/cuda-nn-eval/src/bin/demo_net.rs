use std::time::Instant;

use cuda_sys::wrapper::handle::{cuda_device_count, CudaStream, CudnnHandle};
use nn_cuda_eval::net::{NetDefinition, NetEvaluator};

fn main() {
    main_thread()
}

fn main_thread() {
    println!("Cuda device count: {}", cuda_device_count());
    let stream = CudaStream::new(0);
    let handle = CudnnHandle::new(stream);

    let def = NetDefinition {
        tower_depth: 2 * 8,
        channels: 32,
    };

    let batch_size = 5000;
    let mut eval = NetEvaluator::new(handle, def, batch_size);

    let mut data = vec![0.0; batch_size as usize * def.channels as usize * 7 * 7];

    let start = Instant::now();
    let mut prev_print = Instant::now();

    for i in 0..250 {
        eval.eval(&mut data);

        let now = Instant::now();
        if (now - prev_print).as_secs_f32() >= 1.0 {
            println!("{}", i);

            let throughput = (batch_size * i) as f32 / (now - start).as_secs_f32();
            prev_print = now;

            println!("Throughput: {} boards/s", throughput);
        }
    }
}
