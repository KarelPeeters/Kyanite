use sttt::games::ataxx::AtaxxBoard;
use sttt::util::game_stats;

///Test cases from https://github.com/kz04px/libataxx/blob/master/tests/perft.cpp, edited to remove move counters.
#[test]
fn ataxx_perft() {
    let positions = vec![
        ("7/7/7/7/7/7/7 x", vec![1, 0, 0, 0, 0]),
        ("7/7/7/7/7/7/7 o", vec![1, 0, 0, 0, 0]),
        ("x5o/7/7/7/7/7/o5x x", vec![1, 16, 256, 6460, 155888, 4752668]),
        ("x5o/7/7/7/7/7/o5x o", vec![1, 16, 256, 6460, 155888, 4752668]),
        ("x5o/7/2-1-2/7/2-1-2/7/o5x x", vec![1, 14, 196, 4184, 86528, 2266352]),
        ("x5o/7/2-1-2/7/2-1-2/7/o5x o", vec![1, 14, 196, 4184, 86528, 2266352]),
        ("x5o/7/2-1-2/3-3/2-1-2/7/o5x x", vec![1, 14, 196, 4100, 83104, 2114588]),
        ("x5o/7/2-1-2/3-3/2-1-2/7/o5x o", vec![1, 14, 196, 4100, 83104, 2114588]),
        ("x5o/7/3-3/2-1-2/3-3/7/o5x x", vec![1, 16, 256, 5948, 133264, 3639856]),
        ("x5o/7/3-3/2-1-2/3-3/7/o5x o", vec![1, 16, 256, 5948, 133264, 3639856]),
        ("7/7/7/7/ooooooo/ooooooo/xxxxxxx x", vec![1, 1, 75, 249, 14270, 452980]),
        ("7/7/7/7/ooooooo/ooooooo/xxxxxxx o", vec![1, 75, 249, 14270, 452980]),
        ("7/7/7/7/xxxxxxx/xxxxxxx/ooooooo x", vec![1, 75, 249, 14270, 452980]),
        ("7/7/7/7/xxxxxxx/xxxxxxx/ooooooo o", vec![1, 1, 75, 249, 14270, 452980]),
        ("7/7/7/2x1o2/7/7/7 x", vec![1, 23, 419, 7887, 168317, 4266992]),
        ("7/7/7/2x1o2/7/7/7 o", vec![1, 23, 419, 7887, 168317, 4266992]),
        ("7/7/7/7/-------/-------/x5o x", vec![1, 2, 4, 13, 30, 73, 174]),
        ("7/7/7/7/-------/-------/x5o o", vec![1, 2, 4, 13, 30, 73, 174]),
    ];

    for (fen, expected_perfts) in &positions {
        let board = AtaxxBoard::from_fen(fen);

        for (depth, &expected_perft) in expected_perfts.iter().enumerate() {
            let perft = game_stats::perft(&board, depth as u32);
            println!("Board {}, depth {} -> {}", fen, depth, perft);
            assert_eq!(expected_perft, perft)
        }
    }
}