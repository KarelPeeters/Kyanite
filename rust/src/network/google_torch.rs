use std::path::Path;
use std::time::Instant;

use sttt::board::{Board, Coord};
use tch::{CModule, Device, IValue, maybe_init_cuda, Tensor};

use crate::network::{collect_evaluations, encode_google_input, Network, NetworkEvaluation};
use crate::network::torch_utils::{unwrap_ivalue_pair, unwrap_tensor_with_shape};

#[derive(Debug)]
pub struct GoogleTorchNetwork {
    model: CModule,
    device: Device,

    pub pytorch_time: f32,
}

impl GoogleTorchNetwork {
    pub fn load(path: impl AsRef<Path>, device: Device) -> Self {
        //ensure CUDA support isn't "optimized" away by the linker
        maybe_init_cuda();

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

        let (wdls, policies) = unwrap_ivalue_pair(&result);

        let wdls: Vec<f32> = unwrap_tensor_with_shape(wdls, &[batch_size, 3]).into();
        let policies: Vec<f32> = unwrap_tensor_with_shape(policies, &[batch_size, 9, 9]).into();

        collect_evaluations(boards, &wdls, &policies, Coord::yx)
    }
}
