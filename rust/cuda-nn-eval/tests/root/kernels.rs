use bytemuck::{cast_slice, cast_slice_mut};
use itertools::Itertools;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use cuda_nn_eval::{Device, kernels};
use cuda_nn_eval::device_tensor::DeviceTensor;
use cuda_nn_eval::quant::QuantizedStorage;
use cuda_sys::wrapper::handle::CudaStream;
use cuda_sys::wrapper::status::Status;

#[test]
fn gather() {
    let device = Device::new(0);
    let stream = CudaStream::new(device);

    let input_data = (0..128).map(|x| x as f32).collect_vec();
    let indices_data: Vec<i32> = vec![16, 3, 8, 2, 4, 9];
    let mut output_data = vec![0f32; indices_data.len()];

    let input = device.alloc(input_data.len() * 4);
    let indices = device.alloc(indices_data.len() * 4);
    let output = device.alloc(output_data.len() * 4);

    unsafe {
        input.copy_linear_from_host(cast_slice(&input_data));
        indices.copy_linear_from_host(cast_slice(&indices_data));

        kernels::gatherFloat(
            stream.inner(),
            indices_data.len() as i32,
            indices.ptr() as *const _,
            input.ptr() as *const _,
            output.ptr() as *mut _,
        )
            .unwrap();

        output.copy_linear_to_host(cast_slice_mut(&mut output_data));
    }

    println!("{:?}", output_data);
    let expected_output_data = indices_data.iter().map(|&x| x as f32).collect_vec();
    assert_eq!(output_data, expected_output_data)
}

#[test]
fn gather_2d_axis1() {
    for batch_size in [0, 1, 2, 3, 4, 8, 13] {
        for input_size in [1, 2, 3, 4, 128, 129, 1000] {
            for index_count in [0, 1, 2, 3, 63, 64, 65, 127, 128, 129, 1000] {
                gather_2d_axis1_impl(batch_size, input_size, index_count);
            }
        }
    }
}

#[test]
fn gather_chess_shape() {
    gather_2d_axis1_impl(128, 4608, 1880);
}

fn gather_2d_axis1_impl(batch_size: usize, input_size: usize, index_count: usize) {
    println!("Testing input: {}x{}, indices: {}", batch_size, input_size, index_count);

    let device = Device::new(0);
    let stream = CudaStream::new(device);

    let input_data: Vec<f32> = (0..batch_size * input_size).map(|x| -(x as f32)).collect_vec();

    let mut index_rng = StdRng::seed_from_u64(1);
    let indices_data: Vec<f32> = (0..index_count)
        .map(|_| index_rng.gen_range(0..input_size) as f32)
        .collect_vec();

    let mut output_data: Vec<f32> = vec![0f32; batch_size * indices_data.len()];

    let input = device.alloc(input_data.len() * 4);
    let indices = device.alloc(indices_data.len() * 4);
    let output = device.alloc(output_data.len() * 4);

    unsafe {
        input.copy_linear_from_host(cast_slice(&input_data));
        indices.copy_linear_from_host(cast_slice(&indices_data));

        let before = stream.record_new_event();

        kernels::gather2dAxis1FloatFloat(
            stream.inner(),
            batch_size as i32,
            input_size as i32,
            input_size as i32,
            1,
            indices_data.len() as i32,
            input.ptr() as *const f32,
            indices.ptr() as *const f32,
            output.ptr() as *mut f32,
        )
            .unwrap();

        let after = stream.record_new_event();
        stream.synchronize();
        println!("Took {}s", after.time_elapsed_since(&before));

        output.copy_linear_to_host(cast_slice_mut(&mut output_data));
    }

    let expected_output_data = (0..batch_size)
        .flat_map(|n| {
            indices_data
                .iter()
                .map(|&i| input_data[n * input_size + i as usize])
                .collect_vec()
        })
        .collect_vec();

    if output_data != expected_output_data {
        eprintln!("{:?}", output_data);
        eprintln!("{:?}", expected_output_data);

        for n in 0..batch_size {
            for q in 0..index_count {
                let index = indices_data[q];
                let i = n * index_count + q;
                let actual = output_data[i];
                let expected = expected_output_data[i];
                if actual != expected {
                    println!(
                        "({}, {}) -> [{}] : actual {}, expected {}",
                        n, q, index, actual, expected
                    )
                }
            }
        }
    }
}

#[test]
fn quantize() {
    let device = Device::new(0);

    #[rustfmt::skip]
        let input_data = vec![0.0, 1.0, -1.0, 1.1, -1.1, 0.9, 0.8, -500.1, 500.1, 0.999, -0.999, -0.2, 0.2, 0.16];
    #[rustfmt::skip]
        let expected_output_data = vec![0.0, 1.0, -1.0, 1.0, -1.0, 0.8976378, 0.8031496, -1.0, 1.0, 1.0, -1.0, -0.19685039, 0.19685039, 0.15748031];

    let length = input_data.len();
    assert_eq!(expected_output_data.len(), length);

    let mut quant_data: Vec<u8> = vec![0; length];
    let mut output_data: Vec<f32> = vec![0.0; length];

    let input = DeviceTensor::alloc_simple(device, vec![1, length]);
    let quant = QuantizedStorage::alloc(device, length);
    let output = DeviceTensor::alloc_simple(device, vec![1, length]);

    unsafe {
        input.copy_simple_from_host(&input_data);

        quant.quantize_from(&input);
        quant.unquantize_to(&output);

        quant.ptr().copy_linear_to_host(&mut quant_data);
        output.copy_simple_to_host(&mut output_data);
    }

    println!("{:?}", input_data);
    println!("{:?}", quant_data);
    println!("{:?}", output_data);

    let mut any_error = false;

    for i in 0..length {
        if output_data[i] != expected_output_data[i] {
            eprintln!(
                "Mismatch at i={} for input {}, expected {} got {}",
                i, input_data[i], expected_output_data[i], output_data[i],
            );
            any_error = true;
        }
    }

    if any_error {
        panic!("Wrong output");
    }
}
