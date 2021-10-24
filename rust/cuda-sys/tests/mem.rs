use itertools::Itertools;

use cuda_sys::wrapper::handle::Device;
use cuda_sys::wrapper::mem::device::DeviceMem;

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

#[test]
fn mem_strided() {
    let device = Device::new(0);
    let a = DeviceMem::alloc(128, device);
    let b = DeviceMem::alloc(128, device);

    let a_data = (0..128).collect_vec();
    let mut b_data = vec![0; 128];

    unsafe {
        a.copy_from_host(&a_data);
        b.fill_with_byte(0);

        //TODO 3D is terribly unergonomic, but we should be able to get away with 2D for now
        //  otherwise we probably need to write our own copy kernel

        // cudaMemcpy(b.ptr(), a.ptr(), 128, cudaMemcpyKind::cudaMemcpyDeviceToDevice).unwrap();

        b.copy_to_host(&mut b_data);
    }

    println!("{:?}", b_data);
}
