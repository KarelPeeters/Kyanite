use std::time::Instant;

use bytemuck::{cast_slice, cast_slice_mut};

use cuda_sys::wrapper::handle::{CudaStream, Device};
use cuda_sys::wrapper::rtc::args::KernelArgs;
use cuda_sys::wrapper::rtc::core::{CuModule, Dim3};

const SAXPY_SOURCE: &str = r#"

extern "C" { 

    __global__ void saxpy(float a, float *x, float *y, float *out, int n)
    {
        // assert(false);
        
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n) {
            out[tid] = a * x[tid] + y[tid];
        }
    }

}
"#;

#[test]
fn saxpy() {
    let device = Device::new(0);
    let stream = CudaStream::new(device);

    unsafe {
        for _ in 0..10 {
            let start = Instant::now();

            let result = CuModule::from_source(device, SAXPY_SOURCE, None, &[], &Default::default());
            println!("{}", result.log);

            let module = result.module.unwrap();

            let fun_saxpy2 = module.get_function("saxpy").unwrap();

            let a = 1.0f32;
            let x_mem = device.alloc(4);
            let y_mem = device.alloc(4);
            let out_mem = device.alloc(4);
            let len: i32 = 1;

            x_mem.copy_linear_from_host(cast_slice(&[1f32]));
            y_mem.copy_linear_from_host(cast_slice(&[2f32]));

            let mut args = KernelArgs::new();
            args.push(a);
            args.push(x_mem.ptr());
            args.push(y_mem.ptr());
            args.push(out_mem.ptr());
            args.push(len);
            let args = args.finish();

            fun_saxpy2.launch_kernel(Dim3::new(64, 1, 1), Dim3::new(64, 1, 1), 0, &stream, &args);

            stream.synchronize();

            let mut result = vec![0f32];
            out_mem.copy_linear_to_host(cast_slice_mut(&mut result));

            println!("{:?}", result);
            println!("{:?}", start.elapsed());

            assert_eq!(result[0], 3.0);
        }
    }
}
