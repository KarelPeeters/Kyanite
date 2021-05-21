use std::path::Path;

use itertools::Itertools;
use sttt::board::{Board, Coord};
use tch::{CModule, Device, IValue, TchError, Tensor};

#[derive(Debug)]
pub struct Network {
    model: CModule,
}

#[derive(Debug)]
pub struct NetworkEvaluation {
    pub value: f32,
    pub policy: Vec<f32>,
}

impl Network {
    pub fn load(path: impl AsRef<Path>) -> Self {
        let model = CModule::load_on_device(path.as_ref(), Device::Cpu)
            .expect("Failed to load model");
        Network { model }
    }

    pub fn evaluate(&mut self, board: &Board) -> NetworkEvaluation {
        let mut result = self.evaluate_all(&[board.clone()]);
        assert_eq!(result.len(), 1);
        result.pop().unwrap()
    }

    pub fn evaluate_all(&mut self, boards: &[Board]) -> Vec<NetworkEvaluation> {
        let mut mask = Vec::new();
        let mut tiles = Vec::new();
        let mut macros = Vec::new();

        for board in boards {
            mask.extend(Coord::all().map(|c| board.is_available_move(c) as u8 as f32));

            tiles.extend(Coord::all().map(|c| (board.tile(c) == board.next_player) as u8 as f32));
            tiles.extend(Coord::all().map(|c| (board.tile(c) == board.next_player.other()) as u8 as f32));

            macros.extend((0..9).map(|om| (board.macr(om) == board.next_player) as u8 as f32));
            macros.extend((0..9).map(|om| (board.macr(om) == board.next_player.other()) as u8 as f32));
        }

        let batch_size = boards.len() as i64;
        let batch_mask = Tensor::of_slice(&mask).view([batch_size, 9, 9]);
        let batch_tiles = Tensor::of_slice(&tiles).view([batch_size, 2, 9, 9]);
        let batch_macros = Tensor::of_slice(&macros).view([batch_size, 2, 3, 3]);

        let input = [IValue::Tensor(batch_mask), IValue::Tensor(batch_tiles), IValue::Tensor(batch_macros)];
        let result = self.model.forward_is(&input);

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
        let batch_value = unwrap_tensor_with_shape(&output[0], &[batch_size]);
        let batch_policy = unwrap_tensor_with_shape(&output[1], &[batch_size, 81]);

        let batch_value = Vec::<f32>::from(batch_value);
        let mut batch_policy = Vec::<f32>::from(batch_policy);

        boards.iter().enumerate().map(|(i, board)| {
            let range = (81 * i)..(81 * (i + 1));

            mask_and_softmax(&mut batch_policy[range.clone()], board);

            NetworkEvaluation {
                value: batch_value[i],
                policy: batch_policy[range].to_vec(),
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