use std::path::Path;

use itertools::Itertools;
use sttt::board::{Board, Coord};
use tch::{CModule, Device, IValue, maybe_init_cuda, Tensor};

use crate::network::{collect_evaluations, Network, NetworkEvaluation};
use crate::network::torch_utils::{unwrap_ivalue_pair, unwrap_tensor_with_shape};

#[derive(Debug)]
pub struct MixTorchNetwork {
    model: CModule,
    device: Device,
}

impl MixTorchNetwork {
    pub fn load(path: impl AsRef<Path>, device: Device) -> Self {
        maybe_init_cuda();

        let model = CModule::load_on_device(path.as_ref(), device)
            .expect("Failed to load model");
        MixTorchNetwork { model, device }
    }
}

impl Network for MixTorchNetwork {
    fn evaluate_batch(&mut self, boards: &[Board]) -> Vec<NetworkEvaluation> {
        let batch_size = boards.len() as i64;

        let (input_mask, input_board) = encode_input(boards, self.device);
        let input = [IValue::from(input_mask), IValue::from(input_board)];

        let result = self.model.forward_is(&input);
        let (wdls, policies) = unwrap_ivalue_pair(&result);

        let wdls: Vec<f32> = unwrap_tensor_with_shape(wdls, &[batch_size, 3]).into();
        let policies: Vec<f32> = unwrap_tensor_with_shape(policies, &[batch_size, 81]).into();

        collect_evaluations(boards, &wdls, &policies, Coord::o)
    }
}

fn encode_input(boards: &[Board], device: Device) -> (Tensor, Tensor) {
    let batch_size = boards.len() as i64;

    let input_mask = boards.iter().flat_map(|board| {
        Coord::all().map(move |c| board.is_available_move(c) as u8 as f32)
    }).collect_vec();
    let input_mask = Tensor::of_slice(&input_mask).view([batch_size, 9, 9]).to_device(device);

    let mut input_board = vec![];
    for board in boards {
        input_board.extend(Coord::all().map(|c| board.tile(c) == board.next_player));
        input_board.extend((0..9).map(|om| board.macr(om) == board.next_player));

        input_board.extend(Coord::all().map(|c| board.tile(c) == board.next_player.other()));
        input_board.extend((0..9).map(|om| board.macr(om) == board.next_player.other()));
    }
    let input_board = Tensor::of_slice(&input_board).view([batch_size, 2, 10, 9]).to_device(device);

    (input_mask, input_board)
}

