use npyz::npz::NpzArchive;

use cuda_sys::wrapper::handle::{cuda_device_count, CudaStream, CudnnHandle};
use nn_cuda_eval::load::load_net_params;
use nn_cuda_eval::net::{NetEvaluator, ResNetShape};

fn main() {
    main_thread()
}

fn main_thread() {
    println!("Cuda device count: {}", cuda_device_count());
    let stream = CudaStream::new(0);
    let handle = CudnnHandle::new(stream);

    // TODO the output does not always match the PyTorch output!
    //   sometimes it's a perfect WDL match, sometimes it's completely wrong
    //   also check policy and try different outputs
    // TODO write a script to test this, it won't be the last time
    //   maybe write a bunch of input/output triples to a npz file and run over all of them?
    let shape = ResNetShape {
        board_size: 7,
        input_channels: 3,
        policy_channels: 17,

        tower_channels: 16,
        tower_depth: 8,
        wdl_hidden_size: 5,
    };

    let mut npz = NpzArchive::open("../data/derp/basic_res_model/params.npz").unwrap();
    let params = load_net_params(shape, &mut npz, handle.device());

    let batch_size = 1;
    let mut eval = NetEvaluator::new(handle, shape, params, batch_size);

    let input_size = batch_size * shape.input_channels * shape.board_size * shape.board_size;
    let output_wdl_size = batch_size * 3;
    let output_policy_size = batch_size * shape.policy_channels * shape.board_size * shape.board_size;

    let input = vec![0.0; input_size as usize];
    let mut output_wdl = vec![0.0; output_wdl_size as usize];
    let mut output_policy = vec![0.0; output_policy_size as usize];

    eval.eval(&input, &mut output_wdl, &mut output_policy);

    println!("{:?}", output_policy);
    println!("{:?}", output_wdl);

    /*
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
    */
}
