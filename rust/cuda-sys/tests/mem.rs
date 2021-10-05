use cuda_sys::wrapper::handle::Device;
use cuda_sys::wrapper::mem::DeviceMem;

#[test]
fn mem_slice() {
    let device = Device::new(0);

    let mem = DeviceMem::alloc(512, device);

    unsafe {
        let mut a = mem.slice(0, 128);
        let mut b = mem.slice(128, 128);
        let mut c = mem.slice(256, 128);

        a.fill_with_byte(1);
        b.fill_with_byte(2);
        c.copy_from_device(&a);
    }

    let mut output = vec![0u8; 512];
    mem.copy_to_host(&mut output);

    let mut expected_output = vec![];
    expected_output.extend(std::iter::repeat(1).take(128));
    expected_output.extend(std::iter::repeat(2).take(128));
    expected_output.extend(std::iter::repeat(1).take(128));
    expected_output.extend(std::iter::repeat(0).take(128));

    assert_eq!(expected_output, output);
}