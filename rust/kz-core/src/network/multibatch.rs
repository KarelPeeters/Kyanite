use std::borrow::Borrow;
use std::fmt::Debug;

use board_game::board::Board;
use itertools::Itertools;

use crate::network::{Network, ZeroEvaluation};

#[derive(Debug)]
pub struct MultiBatchNetwork<I> {
    networks: Vec<(usize, I)>,
}

impl<I> MultiBatchNetwork<I> {
    pub fn new(networks: Vec<(usize, I)>) -> Self {
        MultiBatchNetwork { networks }
    }

    pub fn build_sizes(sizes: &[usize], mut f: impl FnMut(usize) -> I) -> Self {
        let networks = sizes.iter().map(|&size| (size, f(size))).collect_vec();
        MultiBatchNetwork { networks }
    }
}

impl<I: Debug> MultiBatchNetwork<I> {
    pub fn used_network_index(&self, batch_size: usize) -> usize {
        (0..self.networks.len())
            .filter(|i| self.networks[*i].0 >= batch_size)
            .min_by_key(|i| self.networks[*i].0)
            .unwrap_or_else(|| panic!("No network for batch size {} in {:?}", batch_size, self))
    }

    pub fn used_batch_size(&self, batch_size: usize) -> usize {
        self.networks[self.used_network_index(batch_size)].0
    }
}

impl<B: Board, I: Network<B>> Network<B> for MultiBatchNetwork<I> {
    fn max_batch_size(&self) -> usize {
        self.networks.iter().map(|(size, _)| *size).max().unwrap_or(0)
    }

    fn evaluate_batch(&mut self, boards: &[impl Borrow<B>]) -> Vec<ZeroEvaluation<'static>> {
        let index = self.used_network_index(boards.len());
        let network = &mut self.networks[index].1;

        network.evaluate_batch(boards)
    }
}
