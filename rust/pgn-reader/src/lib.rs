use std::cmp::max;
use std::io::Read;
use std::ops::ControlFlow;
use std::str::FromStr;

pub use buffered_reader;
use buffered_reader::{BufferedReader, Generic};
use internal_iterator::InternalIterator;
use memchr::{memchr, memchr2_iter};

//TODO support escape codes (mostly in headers and values)

#[derive(Debug)]
pub struct PgnReader<R> {
    start_index: usize,
    prev_game_length: usize,
    input: R,
}

#[derive(Debug, Eq, PartialEq)]
pub struct PgnGame<'a> {
    pub start_index: usize,
    pub header: &'a str,
    pub moves: &'a str,
}

#[derive(Debug, Eq, PartialEq)]
pub struct PgnMove<'a> {
    pub mv: &'a str,
    pub comment: Option<&'a str>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PgnResult {
    WinWhite,
    WinBlack,
    Draw,
    Star,
}

// The evaluation (from the %eval comment), always from the white POV.
#[derive(Debug, Copy, Clone)]
pub enum PgnEval {
    MateIn(i32),
    Pawns(f32),
}

#[derive(Debug)]
pub enum Error {
    InvalidPgn(usize),
    UnexpectedEof,
    Io(std::io::Error),
    Utf8(std::str::Utf8Error),
}

impl<R: BufferedReader<()>> PgnReader<R> {
    pub fn new(input: R) -> Self {
        Self {
            start_index: 0,
            prev_game_length: 0,
            input,
        }
    }
}

impl<R: Read + Send + Sync> PgnReader<Generic<R, ()>> {
    pub fn new_generic(input: R) -> Self {
        PgnReader::new(Generic::new(input, None))
    }
}

impl<'a> PgnGame<'a> {
    pub fn header(&self, key: &str) -> Option<&'a str> {
        let mut left = self.header;
        while let Some(start) = left.find(key) {
            assert_ne!(start, 0, "Invalid pgn header");

            let left_bytes = left.as_bytes();
            if left_bytes[start - 1] == b'[' && left_bytes[start + key.len()] == b' ' {
                let block = &left[start + key.len() + 1..];

                // we've found the right key
                let value_start = block.find('"').unwrap() + 1;
                let value_len = block[value_start..].find('"').unwrap();

                return Some(&block[value_start..value_start + value_len]);
            }

            left = &left[start + key.len()..];
        }
        None
    }

    pub fn move_iter(&self) -> MoveIterator<'a> {
        MoveIterator(self.moves)
    }

    pub fn result(&self) -> PgnResult {
        for &(result, result_str) in RESULT_STR {
            if self.moves.ends_with(result_str) {
                return result;
            }
        }
        let end = &self.moves[self.moves.len().saturating_sub(10)..self.moves.len()];
        panic!("Moves string does not end with a result: {:?}", end);
    }
}

pub struct MoveIterator<'a>(&'a str);

impl<'a> InternalIterator for MoveIterator<'a> {
    type Item = PgnMove<'a>;

    fn try_for_each<R, F>(self, mut f: F) -> ControlFlow<R>
    where
        F: FnMut(Self::Item) -> ControlFlow<R>,
    {
        let mut left = self.0;
        let mut curr_mv = None;

        //TODO try to optimize this loop some more
        loop {
            left = left.trim_start();

            // comment/variation
            if left.as_bytes()[0] == b'{' {
                let skip = variation_length(left);

                // report this comment together with the previously parsed move
                if let Some(mv) = curr_mv {
                    let comment = left[1..skip - 1].trim();
                    f(PgnMove {
                        mv,
                        comment: Some(comment),
                    })?;
                    curr_mv = None;
                }

                left = &left[skip..];
                continue;
            } else {
                // report the previously parsed move without any comment
                if let Some(mv) = curr_mv.take() {
                    f(PgnMove { mv, comment: None })?;
                }
            }
            assert!(curr_mv.is_none());

            // outcome?
            for &(_, outcome_str) in RESULT_STR {
                if let Some(rest) = left.strip_prefix(outcome_str) {
                    assert!(rest.is_empty(), "Leftover stuff after outcome: '{}'", rest);
                    return ControlFlow::Continue(());
                }
            }

            // move
            let start = left
                .find(|c: char| !c.is_ascii_digit() && c != '.' && c != ' ')
                .unwrap();
            let len = left[start..].find(' ').unwrap();

            const MOVE_SUFFIX_CHARS: &[char] = &['?', '!'];
            curr_mv = Some(left[start..start + len].trim_end_matches(MOVE_SUFFIX_CHARS));
            left = &left[start + len..];
        }
    }
}

impl<'a> PgnMove<'a> {
    pub fn field(&self, key: &str) -> Option<&'a str> {
        //TODO this only considers non-nested variations/comments, maybe check the pgn spec
        self.comment.and_then(|comment| {
            comment.split("[%").skip(1).find(|s| s.starts_with(key)).map(|s| {
                let s = s[key.len()..].trim();
                let end = s.find(']').unwrap();
                &s[..end]
            })
        })
    }
}

impl FromStr for PgnResult {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        for &(cand, cand_str) in RESULT_STR {
            if cand_str == s {
                return Ok(cand);
            }
        }

        Err(())
    }
}

const RESULT_STR: &[(PgnResult, &str)] = &[
    (PgnResult::WinWhite, "1-0"),
    (PgnResult::WinBlack, "0-1"),
    (PgnResult::Draw, "1/2-1/2"),
    (PgnResult::Star, "*"),
];

fn variation_length(left: &str) -> usize {
    let left_bytes = left.as_bytes();

    let mut depth = 0;
    let mut count = 0;

    for i in memchr2_iter(b'{', b'}', left_bytes) {
        match left_bytes[i] {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            _ => (),
        }

        count = i + 1;
        if depth == 0 {
            break;
        }
    }

    assert_eq!(depth, 0, "Non-matching {{}} found in '{}'", left);
    count
}

// we cannot implement Iterator here since the PgnGame has a lifetime dependant on self
impl<R: BufferedReader<()>> PgnReader<R> {
    pub fn next_game(&mut self) -> Result<Option<PgnGame>, Error> {
        self.input.consume(self.prev_game_length);
        if self.input.eof() {
            return Ok(None);
        }

        // we're now at the start of a new game
        let mut in_header = true;

        let mut header_length = 0;
        let mut moves_length = 0;

        loop {
            let start = header_length + moves_length;
            let line = buffered_read_to_from(&mut self.input, start, b'\n')?;

            match line.last() {
                // reached eof, check if we still have a game after this loop
                None => break,
                // great, we've parsed a full line
                Some(b'\n') => (),
                // we ran into eof before a newline
                Some(_) => return Err(Error::UnexpectedEof),
            }

            if line[0] == b'[' {
                if !in_header {
                    // the next game is starting, handle this next time
                    break;
                }
                header_length += line.len();
            } else {
                in_header = false;
                moves_length += line.len();
            }
        }

        match (header_length, moves_length) {
            (0, 0) => Ok(None),
            (0, _) | (_, 0) => Err(Error::InvalidPgn(self.start_index)),
            (_, _) => {
                let data = self.input.buffer();
                let total_length = header_length + moves_length;
                assert!(data.len() >= total_length);

                let header = std::str::from_utf8(&data[0..header_length])?.trim();
                let moves = std::str::from_utf8(&data[header_length..total_length])?.trim();

                let game = PgnGame {
                    start_index: self.start_index,
                    header,
                    moves,
                };
                self.start_index += total_length;
                self.prev_game_length = total_length;

                Ok(Some(game))
            }
        }
    }
}

const EVAL_PAWNS_TANH_DIV: f32 = 4.0;

impl PgnEval {
    pub fn parse(eval: &str) -> PgnEval {
        if let Some(n) = eval.strip_prefix('#') {
            PgnEval::MateIn(n.parse::<i32>().unwrap())
        } else {
            PgnEval::Pawns(eval.parse::<f32>().unwrap())
        }
    }

    pub fn as_white_win_prob(self) -> f32 {
        let pawns = match self {
            PgnEval::MateIn(n) => n.signum() as f32 * f32::INFINITY,
            PgnEval::Pawns(p) => p,
        };

        ((pawns / EVAL_PAWNS_TANH_DIV).tanh() + 1.0) / 2.0
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<std::str::Utf8Error> for Error {
    fn from(e: std::str::Utf8Error) -> Self {
        Error::Utf8(e)
    }
}

/// Version of [BufferedReader::read_to] that only starts searching from the given index.
/// Implementation based on the linked function.
fn buffered_read_to_from(
    input: &mut impl BufferedReader<()>,
    start: usize,
    terminal: u8,
) -> Result<&[u8], std::io::Error> {
    let mut n = 128;

    let len = loop {
        let data = &input.data(start + n)?[start..];

        if let Some(newline) = memchr(terminal, data) {
            break newline + 1;
        } else if data.len() < n {
            // EOF.
            break data.len();
        } else {
            // Read more data.
            n = max(2 * n, data.len() + 1024);
        }
    };

    Ok(&input.buffer()[start..start + len])
}
