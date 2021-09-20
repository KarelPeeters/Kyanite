use std::borrow::Borrow;

use board_game::board::Board;
use board_game::wdl::WDL;
use internal_iterator::InternalIterator;

use crate::mapping::PolicyMapper;
use crate::network::ZeroEvaluation;

pub fn decode_output<B: Board, P: PolicyMapper<B>>(
    policy_mapper: P,
    boards: &[impl Borrow<B>],
    batch_wdl_logit: &[f32],
    batch_policy_logit: &[f32],
) -> Vec<ZeroEvaluation> {
    let batch_size = boards.len();
    assert_eq!(batch_size * 3, batch_wdl_logit.len());
    assert_eq!(batch_size * P::POLICY_SIZE, batch_policy_logit.len());

    boards.iter().enumerate().map(|(bi, board)| {
        let board = board.borrow();

        //wdl
        let wdl_left = &batch_wdl_logit[3 * bi..];
        let mut wdl = [wdl_left[0], wdl_left[1], wdl_left[2]];
        softmax_in_place(&mut wdl);
        let wdl = WDL { win: wdl[0], draw: wdl[1], loss: wdl[2] };

        //policy
        let policy_logit = &batch_policy_logit[P::POLICY_SIZE * bi..(P::POLICY_SIZE * bi) + P::POLICY_SIZE];
        let mut policy: Vec<f32> = board.available_moves().map(|mv| {
            policy_mapper.move_to_index(board, mv)
                .map_or(1.0, |index| policy_logit[index])
        }).collect();
        softmax_in_place(&mut policy);

        ZeroEvaluation { wdl, policy }
    }).collect()
}

pub fn softmax_in_place(slice: &mut [f32]) {
    let mut sum = 0.0;
    for v in slice.iter_mut() {
        *v = v.exp();
        sum += *v;
    }
    assert!(sum > 0.0, "Softmax input sum must be strictly positive, was {}", sum);
    for v in slice.iter_mut() {
        *v /= sum;
    }
}