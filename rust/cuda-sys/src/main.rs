use cuda_sys::bindings::cudaGetDeviceFlags;
use cuda_sys::wrapper::status::Status;

fn main() {
    let mut result: u32 = 0;
    unsafe {
        cudaGetDeviceFlags(&mut result as *mut _).unwrap();
    }
    println!("{:b}", result);
}