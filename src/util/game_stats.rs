use std::collections::HashMap;
use std::hash::Hash;

use internal_iterator::InternalIterator;

use crate::board::Board;

/// The number of legal positions reachable after `depth` moves, including duplicates.
/// See https://www.chessprogramming.org/Perft.
pub fn perft(start: impl Board + Hash, depth: u32) -> u64 {
    let mut map = HashMap::default();
    perf_recurse(&mut map, start, depth)
}

fn perf_recurse<B: Board + Hash>(map: &mut HashMap<B, u64>, board: B, depth: u32) -> u64 {
    if depth == 0 { return 1; }

    if let Some(&p) = map.get(&board) {
        return p;
    }

    let mut p = 0;
    board.available_moves().for_each(|mv| {
        p += perf_recurse(map, board.clone_and_play(mv), depth - 1);
    });
    map.insert(board, p);
    p
}