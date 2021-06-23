use std::path::Path;
use std::time::Instant;

use itertools::Itertools;
use sttt::board::Board;
use tch::{CModule, Cuda, Device, IValue, TchError, Tensor};
use torch_sys::dummy_cuda_dependency;

use crate::network::{collect_google_output, encode_google_input, Network, NetworkEvaluation};

#[derive(Debug)]
pub struct GoogleTorchNetwork {
    model: CModule,
    device: Device,

    pub pytorch_time: f32,
}

pub fn all_cuda_devices() -> Vec<Device> {
    (0..Cuda::device_count() as usize).map(Device::Cuda).collect_vec()
}

impl GoogleTorchNetwork {
    pub fn load(path: impl AsRef<Path>, device: Device) -> Self {
        //ensure CUDA support isn't "optimized" away by the linker
        unsafe { dummy_cuda_dependency(); }

        let model = CModule::load_on_device(path.as_ref(), device)
            .expect("Failed to load model");
        GoogleTorchNetwork { model, device, pytorch_time: 0.0 }
    }
}

impl Network for GoogleTorchNetwork {
    fn evaluate_batch(&mut self, boards: &[Board]) -> Vec<NetworkEvaluation> {
        let batch_size = boards.len() as i64;

        let input = encode_google_input(boards);
        let input = Tensor::of_slice(&input).view([batch_size, 5, 9, 9]).to_device(self.device);
        let input = [IValue::Tensor(input)];

        let start = Instant::now();
        let result = self.model.forward_is(&input);
        self.pytorch_time += (Instant::now() - start).as_secs_f32();

        let output = match result {
            Ok(IValue::Tuple(output)) => output,
            Ok(value) =>
                panic!("Expected tuple, got {:?}", value),
            //TODO create issue in tch repo to ask for better error printing
            Err(TchError::Torch(error)) =>
                panic!("Failed to call model, torch error:\n{}", error),
            err => {
                err.expect("Failed to call model");
                unreachable!();
            }
        };

        assert_eq!(2, output.len(), "Return value count mismatch");
        let batch_value = unwrap_tensor_with_shape(&output[0], &[batch_size, 1]);
        let batch_policy = unwrap_tensor_with_shape(&output[1], &[batch_size, 81]);

        let batch_values = Vec::<f32>::from(batch_value);
        let batch_policies = Vec::<f32>::from(batch_policy);

        collect_google_output(boards, &batch_values, &batch_policies)
    }
}


fn unwrap_tensor_with_shape<'i>(value: &'i IValue, size: &[i64]) -> &'i Tensor {
    match value {
        IValue::Tensor(tensor) => {
            assert_eq!(size, tensor.size(), "Tensor size mismatch");
            tensor
        }
        _ => panic!("Expected Tensor, got {:?}", value)
    }
}
