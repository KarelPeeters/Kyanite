use board_game::games::ataxx::AtaxxBoard;
use board_game::util::game_stats;

///Test cases from https://github.com/kz04px/libataxx/blob/master/tests/perft.cpp, edited to remove move counters.
#[test]
fn ataxx_perft() {
    let positions = vec![
        ("7/7/7/7/7/7/7 x 0 1", vec![1, 0, 0, 0, 0]),
        ("7/7/7/7/7/7/7 o 0 1", vec![1, 0, 0, 0, 0]),
        ("x5o/7/7/7/7/7/o5x x 0 1", vec![1, 16, 256, 6460, 155888, 4752668]),
        ("x5o/7/7/7/7/7/o5x o 0 1", vec![1, 16, 256, 6460, 155888, 4752668]),
        ("x5o/7/2-1-2/7/2-1-2/7/o5x x 0 1", vec![1, 14, 196, 4184, 86528, 2266352]),
        ("x5o/7/2-1-2/7/2-1-2/7/o5x o 0 1", vec![1, 14, 196, 4184, 86528, 2266352]),
        ("x5o/7/2-1-2/3-3/2-1-2/7/o5x x 0 1", vec![1, 14, 196, 4100, 83104, 2114588]),
        ("x5o/7/2-1-2/3-3/2-1-2/7/o5x o 0 1", vec![1, 14, 196, 4100, 83104, 2114588]),
        ("x5o/7/3-3/2-1-2/3-3/7/o5x x 0 1", vec![1, 16, 256, 5948, 133264, 3639856]),
        ("x5o/7/3-3/2-1-2/3-3/7/o5x o 0 1", vec![1, 16, 256, 5948, 133264, 3639856]),
        ("7/7/7/7/ooooooo/ooooooo/xxxxxxx x 0 1", vec![1, 1, 75, 249, 14270, 452980]),
        ("7/7/7/7/ooooooo/ooooooo/xxxxxxx o 0 1", vec![1, 75, 249, 14270, 452980]),
        ("7/7/7/7/xxxxxxx/xxxxxxx/ooooooo x 0 1", vec![1, 75, 249, 14270, 452980]),
        ("7/7/7/7/xxxxxxx/xxxxxxx/ooooooo o 0 1", vec![1, 1, 75, 249, 14270, 452980]),
        ("7/7/7/2x1o2/7/7/7 x 0 1", vec![1, 23, 419, 7887, 168317, 4266992]),
        ("7/7/7/2x1o2/7/7/7 o 0 1", vec![1, 23, 419, 7887, 168317, 4266992]),
        ("x5o/7/7/7/7/7/o5x x 100 1", vec![1, 0, 0, 0, 0]),
        ("x5o/7/7/7/7/7/o5x o 100 1", vec![1, 0, 0, 0, 0]),
        ("7/7/7/7/-------/-------/x5o x 0 1", vec![1, 2, 4, 13, 30, 73, 174]),
        ("7/7/7/7/-------/-------/x5o o 0 1", vec![1, 2, 4, 13, 30, 73, 174]),
    ];

    for (fen, expected_perfts) in &positions {
        let board = AtaxxBoard::from_fen(fen);
        println!("Parsed {:?} as {:?}", fen, board.to_fen());
        println!("{}", board);

        for (depth, &expected_perft) in expected_perfts.iter().enumerate() {
            let perft = game_stats::perft(&board, depth as u32);
            println!("   depth {} -> {} =? {}", depth, expected_perft, perft);
            assert_eq!(expected_perft, perft)
        }
    }
}