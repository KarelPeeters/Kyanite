use std::time::Instant;

use cuda_sys::wrapper::handle::{cuda_device_count, CudaStream, CudnnHandle};
use nn_cuda_eval::net::{NetEvaluator, ResNetParams, ResNetShape};

fn main() {
    main_thread()
}

fn main_thread() {
    println!("Cuda device count: {}", cuda_device_count());
    let stream = CudaStream::new(0);
    let handle = CudnnHandle::new(stream);

    let shape = ResNetShape {
        board_size: 7,
        input_channels: 3,
        tower_depth: 8,
        tower_channels: 32,
        wdl_hidden_size: 16,
        policy_channels: 17,
    };

    let params = ResNetParams::dummy(shape, handle.device());

    let batch_size = 1000;
    let mut eval = NetEvaluator::new(handle, shape, params, batch_size);

    let input_size = batch_size * shape.input_channels * shape.board_size * shape.board_size;
    let output_wdl_size = batch_size * 3;
    let output_policy_size = batch_size * shape.policy_channels * shape.board_size * shape.board_size;

    let input = vec![0.0; input_size as usize];
    let mut output_wdl = vec![0.0; output_wdl_size as usize];
    let mut output_policy = vec![0.0; output_policy_size as usize];

    let start = Instant::now();
    let mut prev_print = Instant::now();

    for i in 0..1000 {
        eval.eval(&input, &mut output_wdl, &mut output_policy);

        let now = Instant::now();
        if (now - prev_print).as_secs_f32() >= 1.0 {
            println!("{}", i);

            let throughput = (batch_size * i) as f32 / (now - start).as_secs_f32();
            prev_print = now;

            println!("Throughput: {} boards/s", throughput);
        }
    }
}
