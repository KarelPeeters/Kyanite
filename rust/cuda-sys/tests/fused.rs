use bytemuck::{cast_slice, cast_slice_mut};

use cuda_sys::bindings::cudnnActivationMode_t;
use cuda_sys::wrapper::descriptor::{ActivationDescriptor, ConvolutionDescriptor, FilterDescriptor, TensorDescriptor};
use cuda_sys::wrapper::handle::{CudnnHandle, Device};
use cuda_sys::wrapper::operation::{run_conv_bias_res_activation, STANDARD_CONV_ALGO};

#[test]
fn fused_all() {
    let device = Device::new(0);
    let mut handle = CudnnHandle::new(device);
    let activation_desc = ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY, 0.0);
    let conv_desc = ConvolutionDescriptor::new(0, 0, 1, 1, 1, 1);

    let algo = STANDARD_CONV_ALGO;
    let filter_desc = FilterDescriptor::new(1, 1, 1, 1);
    let filter_mem = device.alloc(filter_desc.size_bytes());
    let io_desc = TensorDescriptor::new(vec![1, 1, 1, 1], vec![1, 1, 1, 1]);
    let input_mem = device.alloc(io_desc.size_bytes());
    let res_mem = device.alloc(io_desc.size_bytes());
    let bias_desc = TensorDescriptor::new(vec![1, 1, 1, 1], vec![1, 1, 1, 1]);
    let bias_mem = device.alloc(bias_desc.size_bytes());
    let mut output_mem = device.alloc(io_desc.size_bytes());

    let work_size = conv_desc.workspace_size(&mut handle, algo, &io_desc, &filter_desc, &io_desc);
    println!("work size: {}", work_size);
    let mut work_ptr = device.alloc(work_size);

    unsafe {
        input_mem.copy_linear_from_host(cast_slice(&[-1f32]));
        bias_mem.copy_linear_from_host(cast_slice(&[2f32]));
        res_mem.copy_linear_from_host(cast_slice(&[3f32]));
        filter_mem.copy_linear_from_host(cast_slice(&[10f32]));

        run_conv_bias_res_activation(
            &mut handle,
            &activation_desc,
            &conv_desc,
            algo,
            &mut work_ptr,
            work_size,
            &filter_desc,
            &filter_mem,
            &io_desc,
            &input_mem,
            Some(&res_mem),
            &bias_desc,
            &bias_mem,
            &io_desc,
            &mut output_mem,
        );

        let mut output = vec![0f32];
        output_mem.copy_linear_to_host(cast_slice_mut(&mut output));

        println!("{:?}", output);
        assert_eq!(output, vec![-5.0]);
    }
}
