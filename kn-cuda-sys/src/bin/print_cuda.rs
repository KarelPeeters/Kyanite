use kn_cuda_sys::wrapper::handle::CudaDevice;

fn main() {
    for device in CudaDevice::all() {
        let cap = device.compute_capability();
        let prop = device.properties();

        println!("{:?}", cap);
        println!("{:#?}", prop);
    }
}
