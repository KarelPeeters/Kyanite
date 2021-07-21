use std::path::Path;

use internal_iterator::InternalIterator;
use board_game::board::BoardAvailableMoves;
use board_game::games::ataxx::{AtaxxBoard, Coord, Move};
use board_game::wdl::WDL;
use tch::{CModule, Device, IValue, maybe_init_cuda, Tensor};

use crate::games::ataxx_output::FROM_DX_DY;
use crate::network::{Network, softmax};
use crate::network::torch_utils::{unwrap_ivalue_pair, unwrap_tensor_with_shape};
use crate::selfplay::generate_zero::NetworkSettings;
use crate::zero::ZeroEvaluation;

#[derive(Debug)]
pub struct AtaxxTorchSettings {
    pub path: String,
    pub devices: Vec<Device>,
    pub threads_per_device: usize,
}

// TODO is it possible to reduce this boilerplate? wait until we're doing STTT again and see what that looks like
impl NetworkSettings<AtaxxBoard> for AtaxxTorchSettings {
    type ThreadParam = Device;
    type Network = AtaxxTorchNetwork;

    fn load_network(&self, device: Self::ThreadParam) -> Self::Network {
        AtaxxTorchNetwork::load(&self.path, device)
    }

    fn thread_params(&self) -> Vec<Self::ThreadParam> {
        self.devices.repeat(self.threads_per_device)
    }
}

#[derive(Debug)]
pub struct AtaxxTorchNetwork {
    model: CModule,
    device: Device,
}

impl AtaxxTorchNetwork {
    pub fn load(path: impl AsRef<Path>, device: Device) -> Self {
        //ensure CUDA support isn't "optimized" away by the linker
        maybe_init_cuda();

        let model = CModule::load_on_device(path.as_ref(), device)
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

        boards.iter().enumerate().map(|(bi, board)| {
            // wdl
            let mut wdl = batch_wdl_logit[3 * bi..3 * (bi + 1)].to_vec();
            softmax(&mut wdl);
            let wdl = WDL {
                win: wdl[0],
                draw: wdl[1],
                loss: wdl[2],
            };

            // policy
            let policy_start = bi * 17 * 7 * 7;
            let mut policy: Vec<f32> = board.available_moves().map(|mv| {
                match mv {
                    Move::Pass => 1.0,
                    Move::Copy { to } => {
                        let to_index = to.dense_i() as usize;
                        batch_policy_logit[policy_start + to_index]
                    }
                    Move::Jump { from, to } => {
                        let dx = from.x() as i8 - to.x() as i8;
                        let dy = from.y() as i8 - to.y() as i8;
                        let from_index = FROM_DX_DY.iter().position(|&(fdx, fdy)| {
                            fdx == dx && fdy == dy
                        }).unwrap();
                        let to_index = to.dense_i() as usize;
                        batch_policy_logit[policy_start + (1 + from_index) * 7 * 7 + to_index]
                    }
                }
            }).collect();
            softmax(&mut policy);

            ZeroEvaluation { wdl, policy }
        }).collect()
    }
}

fn encode_input(boards: &[AtaxxBoard]) -> Vec<f32> {
    let mut input = Vec::new();

    for board in boards {
        let (next_tiles, other_tiles) = board.tiles_pov();
        input.extend(Coord::all().map(|c| next_tiles.has(c) as u8 as f32));
        input.extend(Coord::all().map(|c| other_tiles.has(c) as u8 as f32));
        input.extend(Coord::all().map(|c| board.gaps().has(c) as u8 as f32));
    }

    input
}
