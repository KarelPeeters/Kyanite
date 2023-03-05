use cuda_sys::wrapper::handle::Device;

fn main() {
    for device in Device::all() {
        let cap = device.compute_capability();
        let prop = device.properties();

        println!("{:?}", cap);
        println!("{:#?}", prop);
    }
}
