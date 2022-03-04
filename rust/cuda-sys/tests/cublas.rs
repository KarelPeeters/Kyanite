use bytemuck::{cast_slice, cast_slice_mut};

use cuda_sys::bindings::cublasOperation_t;
use cuda_sys::wrapper::group::{BatchedMatMulArgs, MatMulArg};
use cuda_sys::wrapper::handle::{CublasHandle, Device};
use cuda_sys::wrapper::mem::device::DeviceMem;

#[test]
fn simple() {
    let device = Device::new(0);
    let handle = CublasHandle::new(device);

    let a = DeviceMem::alloc(3 * 2 * 4, device);
    let b = DeviceMem::alloc(3 * 2 * 4, device);
    let c = DeviceMem::alloc(3 * 4, device);

    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c_data: Vec<f32> = vec![0.0, 0.0, 0.0];
    let expected_result = vec![23.0, 67.0, 127.0];

    let a_mat = MatMulArg {
        mem: a.view(),
        trans: cublasOperation_t::CUBLAS_OP_N,
        ld: 1,
        stride: 2,
    };
    let b_mat = MatMulArg {
        mem: b.view(),
        trans: cublasOperation_t::CUBLAS_OP_N,
        ld: 2,
        stride: 2,
    };
    let c_mat = MatMulArg {
        mem: c.view(),
        trans: cublasOperation_t::CUBLAS_OP_N,
        ld: 1,
        stride: 1,
    };

    let args = BatchedMatMulArgs {
        m: 1,
        n: 1,
        k: 2,
        alpha: 1.0,
        beta: 0.0,
        a: a_mat,
        b: b_mat,
        c: c_mat,
        batch_count: 3,
    };

    unsafe {
        a.copy_from_host(cast_slice(&a_data));
        b.copy_from_host(cast_slice(&b_data));

        args.run(&handle);

        c.copy_to_host(cast_slice_mut(&mut c_data));
    }

    println!("{:?}", c_data);
    assert_eq!(c_data, expected_result);
}
