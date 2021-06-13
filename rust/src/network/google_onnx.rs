use crate::network::{Network, NetworkEvaluation};
use sttt::board::Board;

struct GoogleOnnxNetwork {

}

impl GoogleOnnxNetwork {
    fn load(path: &str) -> GoogleOnnxNetwork {
        todo!()
    }
}

impl Network for GoogleOnnxNetwork {
    fn evaluate_batch(&mut self, boards: &[Board]) -> Vec<NetworkEvaluation> {
        todo!()
    }
}