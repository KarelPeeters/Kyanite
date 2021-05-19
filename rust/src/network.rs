use std::path::Path;

use sttt::board::{Board, Coord};
use tch::{CModule, IValue, Tensor, Device, TchError};
use itertools::Itertools;

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
        let mask = Coord::all()
            .map(|c| board.is_available_move(c) as u8 as f32)
            .collect_vec();
        let mask = Tensor::of_slice(&mask).view([1, 9, 9]);

        let mut tiles = Vec::new();
        tiles.extend(Coord::all().map(|c| (board.tile(c) == board.next_player) as u8 as f32));
        tiles.extend(Coord::all().map(|c| (board.tile(c) == board.next_player.other()) as u8 as f32));
        let tiles = Tensor::of_slice(&tiles).view([2, 9, 9]);

        let mut macros = Vec::new();
        macros.extend((0..9).map(|om| (board.macr(om) == board.next_player) as u8 as f32));
        macros.extend((0..9).map(|om| (board.macr(om) == board.next_player.other()) as u8 as f32));
        let macros = Tensor::of_slice(&macros).view([2, 3, 3]);

        let input = [IValue::Tensor(mask), IValue::Tensor(tiles), IValue::Tensor(macros)];
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
            },
        };

        assert_eq!(2, output.len(), "Return value count mismatch");
        let value = unwrap_tensor_with_shape(&output[0], &[1]);
        let policy = unwrap_tensor_with_shape(&output[1], &[1, 81]);

        let value = Vec::<f32>::from(value)[0];
        let mut policy = Vec::<f32>::from(policy);

        mask_and_softmax(&mut policy, board);

        NetworkEvaluation { value, policy }
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