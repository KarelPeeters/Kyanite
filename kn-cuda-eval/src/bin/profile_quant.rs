use bytemuck::cast_slice;
use itertools::Itertools;

use kn_cuda_eval::device_tensor::DeviceTensor;
use kn_cuda_eval::kernels;
use kn_cuda_eval::quant::QuantizedStorage;
use kn_cuda_sys::wrapper::handle::{CudaStream, Device};
use kn_cuda_sys::wrapper::status::Status;
use kn_graph::dtype::DType;

fn main() {
    unsafe { main_inner() }
}

unsafe fn main_inner() {
    let batch_size = 256;
    let elements = 256 * 8 * 8;

    let device = Device::new(0);
    let stream = CudaStream::new(device);

    let sources = (0..batch_size)
        .map(|_| QuantizedStorage::alloc(device, elements))
        .collect_vec();
    let target = DeviceTensor::alloc_simple(device, vec![batch_size, elements], DType::F32);

    let quantized_pointers = sources.iter().map(|q| q.ptr().ptr() as usize).collect_vec();
    let quantized_pointers_device = device.alloc(8 * batch_size);
    quantized_pointers_device.copy_linear_from_host(cast_slice(&quantized_pointers));

    let start = stream.record_event();
    let iterations = 1000;
    for _ in 0..iterations {
        kernels::unquantize(
            stream.inner(),
            batch_size as i32,
            elements as i32,
            quantized_pointers_device.ptr() as *const *const u8,
            target.ptr().ptr() as *mut f32,
        )
        .unwrap();
    }

    let end = stream.record_event();
    end.synchronize();

    let delta = end.time_elapsed_since(&start);
    let throughput = (batch_size * iterations) as f32 / delta;
    println!("Throughput: {}", throughput);
}
