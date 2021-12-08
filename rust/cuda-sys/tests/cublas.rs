use bytemuck::{cast_slice, cast_slice_mut};

use cuda_sys::bindings::{cublasOperation_t, cublasSgemmStridedBatched};
use cuda_sys::wrapper::handle::{CublasHandle, Device};
use cuda_sys::wrapper::mem::device::DeviceMem;
use cuda_sys::wrapper::status::Status;

#[test]
fn simple() {
    let device = Device::new(0);
    let handle = CublasHandle::new(device);

    let a = DeviceMem::alloc(2 * 4, device);
    let b = DeviceMem::alloc(2 * 4, device);
    let c = DeviceMem::alloc(1 * 4, device);

    let a_data: Vec<f32> = vec![1.0, 2.0];
    let b_data: Vec<f32> = vec![3.0, 4.0];
    let mut c_data: Vec<f32> = vec![0.0];

    unsafe {
        a.copy_from_host(cast_slice(&a_data));
        b.copy_from_host(cast_slice(&b_data));

        cublasSgemmStridedBatched(
            handle.inner(),
            cublasOperation_t::CUBLAS_OP_N,
            cublasOperation_t::CUBLAS_OP_N,
            1,
            1,
            2,
            (&1f32) as *const f32,
            a.ptr() as *const f32,
            1,
            0,
            b.ptr() as *const f32,
            2,
            0,
            (&0f32) as *const f32,
            c.ptr() as *mut f32,
            1,
            0,
            1,
        ).unwrap();

        c.copy_to_host(cast_slice_mut(&mut c_data));
    }

    println!("{:?}", c_data);
    assert_eq!(c_data, vec![11.0]);
}