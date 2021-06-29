use rand::thread_rng;
use sttt::board::Board;
use sttt::util::lower_process_priority;

use sttt_zero::mcts_zero::{zero_build_tree, ZeroSettings};
use sttt_zero::network::google_onnx::GoogleOnnxNetwork;

fn main() {
    lower_process_priority();

    let mut rng = thread_rng();
    let mut network = GoogleOnnxNetwork::load("../data/loop/modest_cont/model_5_epochs.onnx");

    let zero_settings = ZeroSettings::new(2.0, true);

    for _ in 0..16 {
        let tree = zero_build_tree(&Board::new(), 10_000, zero_settings, &mut network, &mut rng);
        // println!("{}", tree.display(100));
        println!("{}, {:?}", tree.value(), tree.best_move());
    }
}
