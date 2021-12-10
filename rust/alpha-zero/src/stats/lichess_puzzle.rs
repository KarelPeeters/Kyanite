use std::io::BufRead;

#[derive(Debug)]
pub struct LichessPuzzle<'a> {
    pub puzzle_id: &'a str,
    pub fen: &'a str,
    pub moves: &'a str,
    pub rating: &'a str,
    pub rating_deviation: &'a str,
    pub popularity: &'a str,
    pub nb_plays: &'a str,
    pub themes: &'a str,
    pub game_url: &'a str,
}

impl<'a> LichessPuzzle<'a> {
    fn from_str(s: &'a str) -> Result<Self, ()> {
        let mut iter = s.split(",");
        let puzzle = LichessPuzzle {
            puzzle_id: iter.next().ok_or(())?,
            fen: iter.next().ok_or(())?,
            moves: iter.next().ok_or(())?,
            rating: iter.next().ok_or(())?,
            rating_deviation: iter.next().ok_or(())?,
            popularity: iter.next().ok_or(())?,
            nb_plays: iter.next().ok_or(())?,
            themes: iter.next().ok_or(())?,
            game_url: iter.next().ok_or(())?,
        };
        if iter.next().is_some() { return Err(()); }

        Ok(puzzle)
    }
}

pub fn for_each_lichess_puzzle(mut read: impl BufRead, mut f: impl FnMut(LichessPuzzle) -> ()) {
    let mut buf = String::new();

    loop {
        buf.clear();
        let result = read.read_line(&mut buf).unwrap();
        if result == 0 { break; }

        let line = buf.trim();
        if line.len() > 0 {
            let puzzle = LichessPuzzle::from_str(line).expect("Failed to parse puzzle");
            f(puzzle)
        }
    }
}