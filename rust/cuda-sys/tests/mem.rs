use bytemuck::{cast_slice, cast_slice_mut};
use itertools::Itertools;

use cuda_sys::bindings::{cudaMemcpy2D, cudaMemcpyKind};
use cuda_sys::wrapper::handle::Device;
use cuda_sys::wrapper::mem::DeviceMem;
use cuda_sys::wrapper::status::Status;

#[test]
fn mem_slice() {
    let device = Device::new(0);

    let mem = DeviceMem::alloc(512, device);
    let a = mem.slice(0, 128);
    let b = mem.slice(128, 128);
    let c = mem.slice(256, 128);

    unsafe {
        a.fill_with_byte(1);
        b.fill_with_byte(2);
        c.copy_from_device(&a);
    }

    let mut output = vec![0u8; 512];
    unsafe {
        mem.copy_to_host(&mut output);
    }

    let mut expected_output = vec![];
    expected_output.extend(std::iter::repeat(1).take(128));
    expected_output.extend(std::iter::repeat(2).take(128));
    expected_output.extend(std::iter::repeat(1).take(128));
    expected_output.extend(std::iter::repeat(0).take(128));

    assert_eq!(expected_output, output);
}