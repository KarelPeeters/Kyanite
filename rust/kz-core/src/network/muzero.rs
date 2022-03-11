use std::marker::PhantomData;

use board_game::board::Board;

use nn_graph::graph::Graph;
use nn_graph::optimizer::{optimize_graph, OptimizerSettings};
use nn_graph::shape;
use nn_graph::shape::{Shape, Size};

use crate::mapping::BoardMapper;

pub struct MuZeroGraphs<B: Board, M: BoardMapper<B>> {
    pub mapper: M,
    pub representation: Graph,
    pub dynamics: Graph,
    pub prediction: Graph,

    pub ph: PhantomData<B>,
}

pub struct MuZeroFusedGraphs<B: Board, M: BoardMapper<B>> {
    pub mapper: M,
    pub state_shape: Shape,

    pub root: Graph,
    pub expand: Graph,

    pub ph: PhantomData<B>,
}

impl<B: Board, M: BoardMapper<B>> MuZeroGraphs<B, M> {
    pub fn optimize(&self, settings: OptimizerSettings) -> MuZeroGraphs<B, M> {
        MuZeroGraphs {
            mapper: self.mapper,
            representation: optimize_graph(&self.representation, settings),
            dynamics: optimize_graph(&self.dynamics, settings),
            prediction: optimize_graph(&self.prediction, settings),
            ph: Default::default(),
        }
    }

    pub fn fuse(&self, settings: OptimizerSettings) -> MuZeroFusedGraphs<B, M> {
        let state_shape;
        let root = {
            let mut root = Graph::new();

            let [c, w, h] = self.mapper.input_full_shape();
            let input = root.input(shape![Size::BATCH, c, w, h]);

            let state = root.call(&self.representation, &[input]);
            assert_eq!(state.len(), 1);
            let state = state[0];
            state_shape = root[state].shape.clone();

            let outputs = root.call(&self.prediction, &[state]);
            assert_eq!(outputs.len(), 2);
            root.output_all(&[state, outputs[0], outputs[1]]);

            root
        };

        let expand = {
            let mut expand = Graph::new();

            let [c, w, h] = self.mapper.mv_full_shape();

            let prev_state = expand.input(state_shape.clone());
            let mv = expand.input(shape![Size::BATCH, c, w, h]);

            let state = expand.call(&self.dynamics, &[prev_state, mv]);
            assert_eq!(state.len(), 1);
            let state = state[0];

            let outputs = expand.call(&self.prediction, &[state]);
            assert_eq!(outputs.len(), 2);
            expand.output_all(&[state, outputs[0], outputs[1]]);

            expand
        };

        MuZeroFusedGraphs {
            mapper: self.mapper,
            state_shape,
            root: optimize_graph(&root, settings),
            expand: optimize_graph(&expand, settings),
            ph: PhantomData,
        }
    }
}
