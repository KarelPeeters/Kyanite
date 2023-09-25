use bytemuck::{cast_slice, cast_slice_mut};
use rand::rngs::StdRng;
use rand::{Fill, SeedableRng};

use kn_cuda_eval::kernels;
use kn_cuda_sys::wrapper::handle::{CudaStream, Device};
use kn_cuda_sys::wrapper::status::Status;
use kn_graph::cpu::convolution;
use kn_graph::graph::ConvDetails;
use kn_graph::ndarray::{azip, Array4, ArrayView4};
use kn_graph::shape::Size;

fn main() {
    let batch_size = 1024;

    let io_size = 8;
    let kernel_size = 3;
    let padding = 1;

    let channels = 16;

    let details = ConvDetails {
        input_channels: channels,
        output_channels: channels,
        input_h: io_size,
        input_w: io_size,
        kernel_h: kernel_size,
        kernel_w: kernel_size,
        stride_y: 1,
        stride_x: 1,
        padding_y: padding,
        padding_x: padding,
        output_h: io_size,
        output_w: io_size,
        batch_size: Size::BATCH,
    };

    let mut input = Array4::zeros((batch_size, details.input_channels, io_size, io_size));
    let mut filter = Array4::zeros((
        details.output_channels,
        details.input_channels,
        kernel_size,
        kernel_size,
    ));

    let mut rng = StdRng::seed_from_u64(456);
    input.as_slice_mut().unwrap().try_fill(&mut rng).unwrap();
    filter.as_slice_mut().unwrap().try_fill(&mut rng).unwrap();

    let expected_output = convolution(details, input.view(), filter.view());

    let actual_output = unsafe { test_custom_conv(channels, io_size, kernel_size, input.view(), filter.view()) };

    if false {
        for bi in 0..batch_size {
            println!("bi={}", bi);
            for ki in 0..details.output_channels {
                println!("  ki={}", ki);
                for y in 0..io_size {
                    print!("  ");

                    for x in 0..io_size {
                        print!("{: >6.2},", expected_output[[bi, ki, y, x]]);
                    }

                    print!("    |    ");

                    for x in 0..io_size {
                        print!("{: >6.2},", actual_output[[bi, ki, y, x]])
                    }

                    println!();
                }
                println!();
            }
            println!();
        }
    }

    let mut max_error: f32 = 0.0;
    azip!((&a in &actual_output, &e in &expected_output) {
        max_error = max_error.max((a - e).abs());
    });
    println!("Max error: {}", max_error);
}

unsafe fn test_custom_conv(
    channels: usize,
    io_size: usize,
    kernel_size: usize,
    input: ArrayView4<f32>,
    filter: ArrayView4<f32>,
) -> Array4<f32> {
    assert_eq!(io_size, 8);
    assert_eq!(kernel_size, 3);

    let batch_size = input.shape()[0];
    let output_len = batch_size * channels * io_size * io_size;

    let warmup_iter = 4;
    let bench_iter = 10;

    let device = Device::new(0);

    let input_mem = device.alloc(input.len() * 4);
    let filter_mem = device.alloc(filter.len() * 4);
    let output_mem = device.alloc(output_len * 4);

    input_mem.copy_linear_from_host(cast_slice(input.as_slice().unwrap()));
    filter_mem.copy_linear_from_host(cast_slice(filter.as_slice().unwrap()));
    output_mem.copy_linear_from_host(cast_slice(&vec![0.0; output_len]));

    let launch = |stream: &mut CudaStream| {
        kernels::conv8x3Float(
            stream.inner(),
            batch_size as i32,
            channels as i32,
            channels as i32,
            input_mem.ptr() as *const f32,
            filter_mem.ptr() as *const f32,
            output_mem.ptr() as *mut f32,
        )
        .unwrap()
    };

    let mut stream = CudaStream::new(device);

    let warmup = stream.record_event();
    for _ in 0..warmup_iter {
        launch(&mut stream);
    }
    let start = stream.record_event();
    for _ in 0..bench_iter {
        launch(&mut stream);
    }
    let end = stream.record_event();

    stream.synchronize();

    let mut output_vec = vec![0f32; output_len];
    output_mem.copy_linear_to_host(cast_slice_mut(&mut output_vec));

    let warmup_delta = start.time_elapsed_since(&warmup) / warmup_iter as f32;
    let bench_delta = end.time_elapsed_since(&start) / bench_iter as f32;

    println!("Warmup:");
    println!("  {} s/iter", warmup_delta);
    println!("  {} evals/s", batch_size as f32 / warmup_delta);
    println!("Bench:");
    println!("  {} s/iter", bench_delta);
    println!("  {} evals/s", batch_size as f32 / bench_delta);

    Array4::from_shape_vec((batch_size, channels, io_size, io_size), output_vec).unwrap()
}
