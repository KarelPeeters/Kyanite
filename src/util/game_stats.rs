use std::collections::HashMap;
use std::hash::Hash;

use internal_iterator::InternalIterator;

use crate::board::Board;

/// The number of legal positions reachable after `depth` moves, including duplicates.
/// See https://www.chessprogramming.org/Perft.
pub fn perft<B: Board>(board: &B, depth: u32) -> u64 {
    let mut map = HashMap::default();
    perft_recuse(&mut map, board.clone(), depth)
}

fn perft_recuse<B: Board + Hash>(map: &mut HashMap<(B, u32), u64>, board: B, depth: u32) -> u64 {
    if depth == 0 { return 1; }
    if board.is_done() { return 0; }

    // we need keys (B, depth) because otherwise we risk miscounting if the same board is encountered at different depths
    let key = (board, depth);
    let board = &key.0;

    if let Some(&p) = map.get(&key) {
        return p;
    }

    let mut p = 0;
    board.available_moves().for_each(|mv: B::Move| {
        p += perft_recuse(map, board.clone_and_play(mv), depth - 1);
    });

    map.insert(key, p);
    return p;
}