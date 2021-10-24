use bytemuck::{cast_slice, cast_slice_mut};
use itertools::Itertools;

use cuda_nn_eval::{Device, kernels};
use cuda_sys::wrapper::event::CudaEvent;
use cuda_sys::wrapper::handle::CudaStream;
use cuda_sys::wrapper::mem::device::DeviceMem;
use cuda_sys::wrapper::status::Status;

#[test]
fn strided_copy() {
    let device = Device::new(0);
    let stream = CudaStream::new(device);

    let input_data = (0..128).map(|x| x as f32).collect_vec();
    let mut output_data = vec![0f32; 128];

    let input = DeviceMem::alloc(input_data.len() * 4, device);
    let output = DeviceMem::alloc(output_data.len() * 4, device);

    let rank = 4;
    let size = 56;
    let input_strides: Vec<i32> = vec![64, 8, 0, 2];
    let output_strides: Vec<i32> = vec![24, 8, 4, 1];
    let dense_strides: Vec<i32> = vec![24, 8, 4, 1];

    let start_event = CudaEvent::new();
    let end_event = CudaEvent::new();

    unsafe {
        input.copy_from_host(cast_slice(&input_data));

        stream.record_event(&start_event);

        for _ in 0..128 {
            kernels::stridedCopyFloat(
                stream.inner(),
                rank,
                size,
                input_strides.as_ptr(), output_strides.as_ptr(), dense_strides.as_ptr(),
                input.ptr() as *const f32, output.ptr() as *mut f32,
            ).unwrap();
        }
        stream.record_event(&end_event);

        output.copy_to_host(cast_slice_mut(&mut output_data));
    }

    let delta = end_event.time_elapsed_since(&start_event);
    println!("Copy took {}", delta);
    println!("{:?}", output_data);
}

#[test]
fn gather() {
    let device = Device::new(0);
    let stream = CudaStream::new(device);

    let input_data = (0..128).map(|x| x as f32).collect_vec();
    let indices_data: Vec<i32> = vec![16, 3, 8, 2, 4, 9];
    let mut output_data = vec![0f32; indices_data.len()];

    let input = DeviceMem::alloc(input_data.len() * 4, device);
    let indices = DeviceMem::alloc(indices_data.len() * 4, device);
    let output = DeviceMem::alloc(output_data.len() * 4, device);

    unsafe {
        input.copy_from_host(cast_slice(&input_data));
        indices.copy_from_host(cast_slice(&indices_data));

        kernels::gatherFloat(
            stream.inner(), indices_data.len() as i32,
            indices.ptr() as *const _, input.ptr() as *const _, output.ptr() as *mut _,
        ).unwrap();

        output.copy_to_host(cast_slice_mut(&mut output_data));
    }

    println!("{:?}", output_data);
    let expected_output_data = indices_data.iter().map(|&x| x as f32).collect_vec();
    assert_eq!(output_data, expected_output_data)
}