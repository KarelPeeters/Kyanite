use kn_cuda_sys::wrapper::handle::CudaDevice;

#[test]
fn mem_slice() {
    let device = CudaDevice::new(0);

    let ptr = device.alloc(512);
    let a = ptr.clone().offset_bytes(0);
    let b = ptr.clone().offset_bytes(128);
    let c = ptr.clone().offset_bytes(256);

    unsafe {
        a.copy_linear_from_host(&vec![0; 512]);
        a.copy_linear_from_host(&vec![1; 128]);
        b.copy_linear_from_host(&vec![2; 128]);
        c.copy_linear_from_device(&a, 128);
    }

    let mut output = vec![0u8; 512];
    unsafe {
        ptr.copy_linear_to_host(&mut output);
    }

    //TODO wtf? why?
    let mut expected_output = vec![];
    expected_output.extend(std::iter::repeat(1).take(128));
    expected_output.extend(std::iter::repeat(2).take(128));
    expected_output.extend(std::iter::repeat(1).take(128));
    expected_output.extend(std::iter::repeat(0).take(128));

    assert_eq!(expected_output, output);
}
