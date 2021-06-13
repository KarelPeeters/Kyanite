use sttt::board::Board;

pub mod google_torch;
pub mod google_onnx;

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