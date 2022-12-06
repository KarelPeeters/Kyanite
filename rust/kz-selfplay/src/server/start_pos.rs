use std::ops::RangeInclusive;
use std::sync::Arc;

use board_game::board::{BoardSymmetry, Player};
use board_game::games::ataxx::AtaxxBoard;
use board_game::symmetry::SymmetryDistribution;
use board_game::util::bitboard::BitBoard8;
use itertools::Itertools;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;

pub fn ataxx_start_pos(
    size: u8,
    start_pos: &str,
) -> impl Fn(&mut StdRng) -> AtaxxBoard + Send + Sync + Clone + 'static {
    let mut options: Vec<(f32, Box<dyn Fn(&mut StdRng) -> AtaxxBoard + Send + Sync>)> = vec![];

    match start_pos {
        "default" => {
            options.push((1.0, Box::new(move |_| AtaxxBoard::diagonal(size))));
        }
        "random-gaps-v1" => {
            options.push((
                1.0,
                Box::new(move |rng| AtaxxBoard::diagonal(size).map(SymmetryDistribution.sample(rng))),
            ));
            options.push((0.9, Box::new(move |rng| ataxx_gen_gap_board(rng, size, 0.0..=0.4))));
            options.push((0.1, Box::new(move |rng| ataxx_gen_gap_board(rng, size, 0.4..=1.0))));
        }
        _ => panic!("Unknown ataxx start_pos specification '{start_pos}'"),
    }

    let weighed_index = WeightedIndex::new(options.iter().map(|x| x.0)).unwrap();
    let options = Arc::new(options);

    move |rng| {
        let index = weighed_index.sample(rng);
        (options[index].1)(rng)
    }
}

pub fn ataxx_gen_gap_board(rng: &mut impl Rng, size: u8, gap_range: RangeInclusive<f32>) -> AtaxxBoard {
    let size = size as usize;

    let tile_count_a = 2.clamp(0, (size * size) / 2);
    let tile_count_b = 2.clamp(0, (size * size) / 2);
    let max_gap_count = (size * size) - tile_count_a - tile_count_b;

    let gap_count_start = (max_gap_count as f32 * gap_range.start()) as usize;
    let gap_count_end = (max_gap_count as f32 * gap_range.end()) as usize;
    let gap_count = rng.gen_range(gap_count_start..=gap_count_end);

    let mut all_coords = BitBoard8::FULL_FOR_SIZE[size].into_iter().collect_vec();
    let (sampled, _) = all_coords.partial_shuffle(rng, tile_count_a + tile_count_b + gap_count);

    let tiles_a = BitBoard8::from_coords(sampled[..tile_count_a].iter().copied());
    let tiles_b = BitBoard8::from_coords(sampled[tile_count_a..][..tile_count_b].iter().copied());
    let gaps = BitBoard8::from_coords(sampled[tile_count_a + tile_count_b..].iter().copied());

    let next_player = if rng.gen() { Player::A } else { Player::B };
    let board = AtaxxBoard::from_parts(size as u8, tiles_a, tiles_b, gaps, 0, next_player);

    board
}
