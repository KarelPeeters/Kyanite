use cuda_sys::bindings::cudnnActivationMode_t;
use cuda_sys::wrapper::descriptor::{ActivationDescriptor, ConvolutionDescriptor, FilterDescriptor, TensorDescriptor};
use cuda_sys::wrapper::group::FusedConvolutionArgs;
use cuda_sys::wrapper::handle::{CudnnHandle, Device};
use cuda_sys::wrapper::mem::device::DeviceMem;
use cuda_sys::wrapper::operation::STANDARD_CONV_ALGO;

fn main() {
    let device = Device::new(0);

    let handle0 = CudnnHandle::new(device);
    let handle1 = CudnnHandle::new(device);

    let algo = STANDARD_CONV_ALGO;

    let channels: i32 = 16;
    let conv_desc = ConvolutionDescriptor::new(1, 1, 1, 1, 1, 1);
    let input_desc = TensorDescriptor::new(vec![128, channels, 8, 8], vec![channels * 8 * 8, 8 * 8, 8, 1]);
    let filter_desc = FilterDescriptor::new(channels, channels, 3, 3);
    let output_desc = TensorDescriptor::new(vec![128, channels, 8, 8], vec![channels * 8 * 8, 8 * 8, 8, 1]);
    let bias_desc = TensorDescriptor::new(vec![1, channels, 1, 1], vec![channels, 1, 1, 1]);

    let work_size = conv_desc.workspace_size(&handle0, algo, &input_desc, &filter_desc, &output_desc);
    dbg!(work_size);

    let channels = channels as usize;
    let conv0 = FusedConvolutionArgs {
        conv_desc,
        algo,
        work_mem: DeviceMem::alloc(work_size, device),
        filter_desc,
        filter_mem: DeviceMem::alloc(channels * channels * 3 * 3 * 4, device),
        input_desc,
        input_mem: DeviceMem::alloc(128 * channels * 8 * 8 * 4, device),
        res_mem: None,
        bias_desc,
        bias_mem: DeviceMem::alloc(channels * 4, device),
        act_desc: ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY, 0.0),
        output_desc,
        output_mem: DeviceMem::alloc(128 * channels * 8 * 8 * 4, device),
    };

    let channels = channels as i32;
    let conv_desc = ConvolutionDescriptor::new(1, 1, 1, 1, 1, 1);
    let input_desc = TensorDescriptor::new(vec![128, channels, 8, 8], vec![channels * 8 * 8, 8 * 8, 8, 1]);
    let filter_desc = FilterDescriptor::new(channels, channels, 3, 3);
    let output_desc = TensorDescriptor::new(vec![128, channels, 8, 8], vec![channels * 8 * 8, 8 * 8, 8, 1]);
    let bias_desc = TensorDescriptor::new(vec![1, channels, 1, 1], vec![channels, 1, 1, 1]);

    let channels = channels as usize;
    let conv1 = FusedConvolutionArgs {
        conv_desc,
        algo,
        work_mem: DeviceMem::alloc(work_size, device),
        filter_desc,
        filter_mem: DeviceMem::alloc(channels * channels * 3 * 3 * 4, device),
        input_desc,
        input_mem: DeviceMem::alloc(128 * channels * 8 * 8 * 4, device),
        res_mem: None,
        bias_desc,
        bias_mem: DeviceMem::alloc(channels * 4, device),
        act_desc: ActivationDescriptor::new(cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY, 0.0),
        output_desc,
        output_mem: DeviceMem::alloc(128 * channels * 8 * 8 * 4, device),
    };

    let par = true;

    unsafe {
        let stream0 = handle0.stream();
        let stream1 = handle1.stream();

        for _ in 0..100 {
            let start0 = stream0.record_new_event();
            stream1.wait_for_event(&start0);

            for _ in 0..1000 {
                conv0.run(&handle0);
                conv1.run(if par { &handle1 } else { &handle0 });
            }

            let end1 = stream1.record_new_event();
            stream0.wait_for_event(&end1);
            let end0 = stream0.record_new_event();
            end0.synchronize();

            println!("Took {:.4}ms", end0.time_elapsed_since(&start0));
        }
    }
}