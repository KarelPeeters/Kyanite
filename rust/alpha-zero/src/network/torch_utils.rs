use itertools::Itertools;
use tch::{Cuda, Device, IValue, TchError, Tensor};

pub fn all_cuda_devices() -> Vec<Device> {
    (0..Cuda::device_count() as usize).map(Device::Cuda).collect_vec()
}

pub fn unwrap_tensor_with_shape<'i>(value: &'i IValue, size: &[i64]) -> &'i Tensor {
    match value {
        IValue::Tensor(tensor) => {
            assert_eq!(size, tensor.size(), "Tensor size mismatch");
            tensor
        }
        _ => panic!("Expected Tensor, got {:?}", value)
    }
}

//TODO create issue in tch repo to ask for better error printing
pub fn unwrap_ivalue_pair(result: &Result<IValue, TchError>) -> (&IValue, &IValue) {
    let output = match result {
        Ok(IValue::Tuple(output)) => output,
        Ok(value) =>
            panic!("Expected tuple, got {:?}", value),
        Err(TchError::Torch(error)) =>
            panic!("Failed to call model, torch error:\n{}", error),
        err => {
            err.as_ref().expect("Failed to call model");
            unreachable!();
        }
    };

    assert_eq!(2, output.len(), "Return value count mismatch");
    (&output[0], &output[1])
}