use cuda_nn_eval::device_tensor::DeviceTensor;
use cuda_nn_eval::quant::QuantizedStorage;
use cuda_nn_eval::Device;

#[test]
fn quantize() {
    let device = Device::new(0);

    #[rustfmt::skip]
        let input_data = vec![0.0, 1.0, -1.0, 1.1, -1.1, 0.9, 0.8, -500.1, 500.1, 0.999, -0.999, -0.2, 0.2, 0.16];
    #[rustfmt::skip]
        let expected_output_data = vec![0.0, 1.0, -1.0, 1.0, -1.0, 0.8976378, 0.8031496, -1.0, 1.0, 1.0, -1.0, -0.19685039, 0.19685039, 0.15748031];

    let length = input_data.len();
    assert_eq!(expected_output_data.len(), length);

    let mut quant_data: Vec<u8> = vec![0; length];
    let mut output_data: Vec<f32> = vec![0.0; length];

    let input = DeviceTensor::alloc_simple(device, vec![1, length]);
    let quant = QuantizedStorage::alloc(device, length);
    let output = DeviceTensor::alloc_simple(device, vec![1, length]);

    unsafe {
        input.copy_simple_from_host(&input_data);

        quant.quantize_from(&input);
        quant.unquantize_to(&output);

        quant.ptr().copy_linear_to_host(&mut quant_data);
        output.copy_simple_to_host(&mut output_data);
    }

    println!("{:?}", input_data);
    println!("{:?}", quant_data);
    println!("{:?}", output_data);

    let mut any_error = false;

    for i in 0..length {
        if output_data[i] != expected_output_data[i] {
            eprintln!(
                "Mismatch at i={} for input {}, expected {} got {}",
                i, input_data[i], expected_output_data[i], output_data[i],
            );
            any_error = true;
        }
    }

    if any_error {
        panic!("Wrong output");
    }
}
