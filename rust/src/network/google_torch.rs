use std::path::Path;

use itertools::Itertools;
use sttt::board::{Board, Coord};
use tch::{CModule, Device, IValue, TchError, Tensor};
use std::time::Instant;

use torch_sys::dummy_cuda_dependency;
use crate::network::{NetworkEvaluation, Network};

#[derive(Debug)]
pub struct GoogleTorchNetwork {
    model: CModule,
    device: Device,

    pub pytorch_time: f32,
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
        let mut input = Vec::new();

        for board in boards {
            input.extend(Coord::all_yx().map(|c| board.is_available_move(c) as u8 as f32));
            input.extend(Coord::all_yx().map(|c| (board.tile(c) == board.next_player) as u8 as f32));
            input.extend(Coord::all_yx().map(|c| (board.tile(c) == board.next_player.other()) as u8 as f32));
            input.extend(Coord::all_yx().map(|c| (board.macr(c.om()) == board.next_player) as u8 as f32));
            input.extend(Coord::all_yx().map(|c| (board.macr(c.om()) == board.next_player.other()) as u8 as f32));
        }

        let batch_size = boards.len() as i64;

        //TODO figure out a way to do copying concurrently
        //  or alternatively figure out a way to compress these tensors:
        //  right now they are 81 * 3 + 9 * 2 = 261 floats = 1044 bytes
        //  while they can easily be represented as 261 bits ~= 32 bytes
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

        let batch_value = Vec::<f32>::from(batch_value);
        let batch_policy = Vec::<f32>::from(batch_policy);

        boards.iter().enumerate().map(|(i, board)| {
            let range = (81 * i)..(81 * (i + 1));
            let policy_yx = &batch_policy[range];
            let mut policy = Coord::all()
                .map(|c| policy_yx[c.yx() as usize])
                .collect_vec();

            mask_and_softmax(&mut policy, board);

            NetworkEvaluation {
                value: batch_value[i],
                policy,
            }
        }).collect_vec()
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

fn mask_and_softmax(slice: &mut [f32], board: &Board) {
    assert_eq!(81, slice.len());

    let mut sum = 0.0;
    for (o, v) in slice.iter_mut().enumerate() {
        if !board.is_available_move(Coord::from_o(o as u8)) {
            *v = 0.0;
        } else {
            *v = v.exp();
            sum += *v;
        }
    }

    for v in slice.iter_mut() {
        *v /= sum;
    }
}