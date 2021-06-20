use sttt::board::{Board, Coord};

pub mod google_torch;

#[derive(Debug)]
pub struct NetworkEvaluation {
    pub value: f32,
    pub policy: Vec<f32>,
}


pub trait Network {
    fn evaluate_batch(&mut self, boards: &[Board]) -> Vec<NetworkEvaluation>;

    fn evaluate(&mut self, board: &Board) -> NetworkEvaluation {
        let mut result = self.evaluate_batch(&[board.clone()]);
        assert_eq!(result.len(), 1);
        result.pop().unwrap()
    }
}

fn encode_google_input(boards: &[Board]) -> Vec<f32> {
    let capacity = boards.len() * 5 * 9 * 9;
    let mut result = Vec::with_capacity(capacity);

    for board in boards {
        result.extend(Coord::all_yx().map(|c| board.is_available_move(c) as u8 as f32));
        result.extend(Coord::all_yx().map(|c| (board.tile(c) == board.next_player) as u8 as f32));
        result.extend(Coord::all_yx().map(|c| (board.tile(c) == board.next_player.other()) as u8 as f32));
        result.extend(Coord::all_yx().map(|c| (board.macr(c.om()) == board.next_player) as u8 as f32));
        result.extend(Coord::all_yx().map(|c| (board.macr(c.om()) == board.next_player.other()) as u8 as f32));
    }

    assert_eq!(capacity, result.len());
    result
}