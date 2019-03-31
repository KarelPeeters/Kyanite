#[cfg(test)]
mod tests {
    use rand::{self, SeedableRng, seq::SliceRandom};
    use rand_xorshift::XorShiftRng;

    use crate::board::{Board, Coord};

    #[test]
    fn test_random_distribution() {
        let mut board = Board::new();
        let mut rand = XorShiftRng::from_seed([0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 1]);

        while !board.is_done() {
            let moves: Vec<Coord> = board.available_moves().collect();

            let mut counts: [i32; 81] = [0; 81];
            for _ in 0..1_000_000 {
                counts[board.random_available_move(&mut rand).unwrap().o() as usize] += 1;
            }

            let avg = (1_000_000 / moves.len()) as i32;

            for (mv, &count) in counts.iter().enumerate() {
                if moves.contains(&Coord::of_o(mv as u8)) {
                    assert!((count.wrapping_sub(avg)).abs() < 10_000, "uniformly distributed")
                } else {
                    assert_eq!(count, 0, "only actual moves returned")
                }
            }

            let mv = moves.choose(&mut rand).unwrap().o();
            board.play(Coord::of_o(mv as u8));
        }
    }
}