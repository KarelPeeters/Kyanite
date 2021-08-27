use std::path::Path;

use tch::{CModule, Device, IValue, maybe_init_cuda, Tensor};

use board_game::games::ataxx::AtaxxBoard;

use crate::games::ataxx_utils::{decode_output, encode_input};
use crate::network::{Network, ZeroEvaluation};
use crate::network::torch_utils::{unwrap_ivalue_pair, unwrap_tensor_with_shape};
use std::ffi::OsStr;

#[derive(Debug)]
pub struct AtaxxTorchNetwork {
    model: CModule,
    device: Device,
}

impl AtaxxTorchNetwork {
    pub fn load(path: impl AsRef<Path>, device: Device) -> Self {
        //ensure CUDA support isn't "optimized" away by the linker
        maybe_init_cuda();

        let path = path.as_ref();
        assert!(path.is_file(), "Trying to load pytorch file {:?} which does not exist", path);
        assert_eq!(Some(OsStr::new("pt")), path.extension(), "Unexpected extension");

        let model = CModule::load_on_device(path, device)
            .expect("Failed to load model");
        AtaxxTorchNetwork { model, device }
    }
}

impl Network<AtaxxBoard> for AtaxxTorchNetwork {
    fn evaluate_batch(&mut self, boards: &[AtaxxBoard]) -> Vec<ZeroEvaluation> {
        let batch_size = boards.len() as i64;

        let input = encode_input(boards);
        let input = Tensor::of_slice(&input).view([batch_size, 3, 7, 7]).to_device(self.device);
        let input = [IValue::Tensor(input)];

        let result = self.model.forward_is(&input);
        let (batch_wdl_logit, batch_policy_logit) = unwrap_ivalue_pair(&result);

        let batch_wdl_logit: Vec<f32> = unwrap_tensor_with_shape(batch_wdl_logit, &[batch_size, 3]).into();
        let batch_policy_logit: Vec<f32> = unwrap_tensor_with_shape(batch_policy_logit, &[batch_size, 17, 7, 7]).into();

        decode_output(boards, &batch_wdl_logit, &batch_policy_logit)
    }
}

