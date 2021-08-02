use serde::{Deserialize, Serialize};
use cuda_nn_eval::graph::Graph;

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct TowerShape {
    pub board_size: i32,
    pub input_channels: i32,

    pub tower_depth: usize,
    pub tower_channels: i32,

    pub wdl_hidden_size: i32,
    pub policy_channels: i32,
}

impl TowerShape {
    pub fn to_graph(&self, batch_size: i32) -> Graph {
        let mut g = Graph::default();

        let n = batch_size;
        let w = self.board_size;
        let c = self.tower_channels;

        let input = g.input([n, self.input_channels, w, w]);

        let tower = g.conv_bias(input, c, 3, 1);
        let mut tower = g.relu(tower);

        for _ in 0..self.tower_depth {
            let first = g.conv_bias(tower, c, 3, 1);
            let first = g.relu(first);

            let second = g.conv_bias(first, c, 3, 1);
            let second = g.add(tower, second);

            tower = g.relu(second)
        }

        let wdl_flat = g.conv_bias(tower, 1, 1, 0);
        let wdl_flat = g.relu(wdl_flat);

        let wdl_hidden = g.flatten_linear_bias(wdl_flat, self.wdl_hidden_size);
        let wdl_hidden = g.relu(wdl_hidden);

        let wdl_output = g.flatten_linear_bias(wdl_hidden, 3);
        g.output(wdl_output);

        let policy_output = g.conv_bias(tower, self.policy_channels, 1, 0);
        g.output(policy_output);

        g
    }
}
