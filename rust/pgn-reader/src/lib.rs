use std::cmp::max;

use buffered_reader::BufferedReader;
use memchr::memchr;

//TODO support escape codes (mostly in headers and values)

#[derive(Debug)]
pub struct PgnReader<R> {
    start_index: usize,
    prev_game_length: usize,
    input: R,
}

#[derive(Debug)]
pub struct Game<'a> {
    start_index: usize,
    pub header: &'a str,
    pub moves: &'a str,
}

#[derive(Debug)]
pub enum Error {
    InvalidPgn(usize),
    UnexpectedEof,
    Io(std::io::Error),
    Utf8(std::str::Utf8Error),
}

impl<R> PgnReader<R> {
    pub fn new(input: R) -> Self {
        Self {
            start_index: 0,
            prev_game_length: 0,
            input,
        }
    }
}

impl<'a> Game<'a> {
    pub fn header(&self, key: &str) -> Option<&str> {
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
}

impl<R: BufferedReader<()>> PgnReader<R> {
    pub fn next(&mut self) -> Result<Option<Game>, Error> {
        self.input.consume(self.prev_game_length);
        if self.input.eof() { return Ok(None); }

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
            (0, 0) =>
                Ok(None),
            (0, _) | (_, 0) =>
                Err(Error::InvalidPgn(self.start_index)),
            (_, _) => {
                let data = self.input.buffer();
                let total_length = header_length + moves_length;
                assert!(data.len() >= total_length);

                let header = std::str::from_utf8(&data[0..header_length])?;
                let moves = std::str::from_utf8(&data[header_length..total_length])?;

                let game = Game { start_index: self.start_index, header, moves };
                self.start_index += total_length;
                self.prev_game_length = total_length;

                Ok(Some(game))
            }
        }
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
fn buffered_read_to_from(input: &mut impl BufferedReader<()>, start: usize, terminal: u8) -> Result<&[u8], std::io::Error> {
    let mut n = 128;

    let len = loop {
        let data = &input.data(start + n)?[start..];

        if let Some(newline) = memchr(terminal, data)
        {
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