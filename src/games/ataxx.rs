use std::cmp::{max, Ordering};
use std::fmt::{Debug, Display, Formatter};
use std::fmt::Write;

use internal_iterator::InternalIterator;
use newtype_ops::newtype_ops;
use rand::Rng;

use crate::board::{Board, BoardAvailableMoves, Outcome, Player};
use crate::symmetry::D4Symmetry;
use crate::util::bits::{BitIter, get_nth_set_bit};

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Coord(u8);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Move {
    Pass,
    Copy { to: Coord },
    Jump { from: Coord, to: Coord },
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct AtaxxBoard {
    tiles_a: Tiles,
    tiles_b: Tiles,
    gaps: Tiles,
    next_player: Player,
    outcome: Option<Outcome>,
}

impl AtaxxBoard {
    pub fn empty() -> Self {
        AtaxxBoard {
            tiles_a: Tiles::empty(),
            tiles_b: Tiles::empty(),
            gaps: Tiles::empty(),
            next_player: Player::A,
            outcome: Some(Outcome::Draw),
        }
    }

    pub fn new_without_blocks() -> Self {
        AtaxxBoard {
            tiles_a: Tiles::CORNERS_A,
            tiles_b: Tiles::CORNERS_B,
            gaps: Tiles::empty(),
            next_player: Player::A,
            outcome: None,
        }
    }

    pub fn tile(&self, coord: Coord) -> Option<Player> {
        if self.tiles_a.has(coord) {
            return Some(Player::A);
        }
        if self.tiles_b.has(coord) {
            return Some(Player::B);
        }
        return None;
    }

    pub fn tiles_a(&self) -> Tiles {
        self.tiles_a
    }

    pub fn tiles_b(&self) -> Tiles {
        self.tiles_b
    }

    pub fn gaps(&self) -> Tiles {
        self.gaps
    }

    pub fn free_tiles(&self) -> Tiles {
        !(self.tiles_a | self.tiles_b | self.gaps)
    }

    /// Return whether tha player with the given tiles has to pass, ie. cannot make a copy or jump move.
    fn must_pass(&self, tiles: Tiles) -> bool {
        let possible_targets = tiles.copy_targets() | tiles.jump_targets();
        (possible_targets & self.free_tiles()).is_empty()
    }

    pub fn tiles_pov(&self) -> (Tiles, Tiles) {
        match self.next_player() {
            Player::A => (self.tiles_a, self.tiles_b),
            Player::B => (self.tiles_b, self.tiles_a),
        }
    }

    fn tiles_pov_mut(&mut self) -> (&mut Tiles, &mut Tiles) {
        match self.next_player {
            Player::A => (&mut self.tiles_a, &mut self.tiles_b),
            Player::B => (&mut self.tiles_b, &mut self.tiles_a),
        }
    }

    /// Set the correct outcome based on the current tiles and gaps.
    fn update_outcome(&mut self) {
        let a_empty = self.tiles_a.is_empty();
        let b_empty = self.tiles_b.is_empty();

        let a_pass = self.must_pass(self.tiles_a);
        let b_pass = self.must_pass(self.tiles_b);

        let outcome = if a_empty && b_empty {
            Some(Outcome::Draw)
        } else if a_empty {
            Some(Outcome::WonBy(Player::B))
        } else if b_empty {
            Some(Outcome::WonBy(Player::A))
        } else if a_pass && b_pass {
            let count_a = self.tiles_a.count();
            let count_b = self.tiles_b.count();

            let outcome = match count_a.cmp(&count_b) {
                Ordering::Less => Outcome::WonBy(Player::B),
                Ordering::Equal => Outcome::Draw,
                Ordering::Greater => Outcome::WonBy(Player::A),
            };
            Some(outcome)
        } else {
            None
        };

        self.outcome = outcome;
    }
}

impl Board for AtaxxBoard {
    type Move = Move;
    type Symmetry = D4Symmetry;

    fn can_lose_after_move() -> bool {
        true
    }

    fn game_length_bounds() -> (u32, Option<u32>) {
        (3, None)
    }

    fn next_player(&self) -> Player {
        self.next_player
    }

    fn is_available_move(&self, mv: Self::Move) -> bool {
        assert!(!self.is_done());

        let next_tiles = self.tiles_pov().0;

        match mv {
            Move::Pass =>
                self.must_pass(next_tiles),
            Move::Copy { to } =>
                (self.free_tiles() & next_tiles.copy_targets()).has(to),
            Move::Jump { from, to } =>
                self.free_tiles().has(to) && next_tiles.has(from) && from.distance(to) == 2,
        }
    }

    fn random_available_move(&self, rng: &mut impl Rng) -> Self::Move {
        assert!(!self.is_done());

        let next_tiles = self.tiles_pov().0;
        let free_tiles = self.free_tiles();

        if self.must_pass(next_tiles) {
            return Move::Pass;
        }

        let copy_targets = self.free_tiles() & next_tiles.copy_targets();
        let jump_targets = free_tiles & next_tiles.jump_targets();

        let copy_count = copy_targets.count() as u32;
        let jump_count: u32 = jump_targets.into_iter().map(|to| {
            (next_tiles & Tiles::coord(to).jump_targets()).count() as u32
        }).sum();

        let index = rng.gen_range(0..(copy_count + jump_count));

        if index < copy_count {
            Move::Copy { to: copy_targets.get_nth(index) }
        } else {
            let mut left = index - copy_count;
            for to in jump_targets {
                let from = next_tiles & Tiles::coord(to).jump_targets();
                let count = from.count() as u32;
                if left < count {
                    let from = from.get_nth(left);
                    return Move::Jump { from, to };
                }
                left -= count;
            }

            unreachable!()
        }
    }

    fn play(&mut self, mv: Self::Move) {
        assert!(self.is_available_move(mv), "{:?} is not available", mv);

        let (next_tiles, other_tiles) = self.tiles_pov_mut();

        let to = match mv {
            Move::Pass => {
                // we don't need to check whether the game is finished now because the other player is guaranteed to have
                //   a real move, since otherwise the game would have finished already
                self.next_player = self.next_player.other();
                return;
            }
            Move::Copy { to } => to,
            Move::Jump { from, to } => {
                *next_tiles &= !Tiles::coord(from);
                to
            }
        };

        let to = Tiles::coord(to);
        let converted = *other_tiles & to.copy_targets();
        *next_tiles |= to | converted;
        *other_tiles &= !converted;

        self.update_outcome();
        self.next_player = self.next_player.other();
    }

    fn outcome(&self) -> Option<Outcome> {
        self.outcome
    }

    fn map(&self, sym: Self::Symmetry) -> Self {
        AtaxxBoard {
            tiles_a: self.tiles_a.map(sym),
            tiles_b: self.tiles_b.map(sym),
            gaps: self.gaps.map(sym),
            next_player: self.next_player,
            outcome: self.outcome,
        }
    }

    fn map_move(sym: Self::Symmetry, mv: Self::Move) -> Self::Move {
        match mv {
            Move::Pass => Move::Pass,
            Move::Copy { to } => Move::Copy { to: to.map(sym) },
            Move::Jump { from, to } => Move::Jump { from: from.map(sym), to: to.map(sym) },
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Tiles(u64);

newtype_ops! { [Tiles] {bitand bitor bitxor} {:=} Self Self }

impl std::ops::Not for Tiles {
    type Output = Tiles;

    fn not(self) -> Self::Output {
        Tiles((!self.0) & Tiles::FULL_MASK)
    }
}

impl IntoIterator for Tiles {
    type Item = Coord;
    type IntoIter = std::iter::Map<BitIter<u64>, fn(u8) -> Coord>;

    fn into_iter(self) -> Self::IntoIter {
        BitIter::new(self.0).map(|i| Coord::from_i(i as u8))
    }
}

impl Tiles {
    pub const FULL_MASK: u64 = 0x7F_7F_7F_7F_7F_7F_7F;
    pub const CORNERS_A: Tiles = Tiles(0x_01_00_00_00_00_00_40);
    pub const CORNERS_B: Tiles = Tiles(0x_40_00_00_00_00_00_01);

    pub fn full() -> Tiles {
        Tiles(Self::FULL_MASK)
    }

    pub fn empty() -> Tiles {
        Tiles(0)
    }

    pub fn coord(coord: Coord) -> Tiles {
        Tiles(1 << coord.i())
    }

    pub fn has(self, coord: Coord) -> bool {
        (self.0 >> coord.i()) & 1 != 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn is_full(self) -> bool {
        self.0 == Self::FULL_MASK
    }

    pub fn count(self) -> u8 {
        self.0.count_ones() as u8
    }

    pub fn get_nth(self, index: u32) -> Coord {
        Coord::from_i(get_nth_set_bit(self.0, index))
    }

    #[must_use]
    pub fn set(self, coord: Coord) -> Self {
        Tiles(self.0 | (1 << coord.i()))
    }

    #[must_use]
    pub fn clear(self, coord: Coord) -> Self {
        Tiles(self.0 & !(1 << coord.i()))
    }

    pub fn left(self) -> Self {
        Tiles((self.0 >> 1) & Self::FULL_MASK)
    }

    pub fn right(self) -> Self {
        Tiles((self.0 << 1) & Self::FULL_MASK)
    }

    pub fn down(self) -> Self {
        Tiles((self.0 >> 8) & Self::FULL_MASK)
    }

    pub fn up(self) -> Self {
        Tiles((self.0 << 8) & Self::FULL_MASK)
    }

    pub fn copy_targets(self) -> Self {
        // counterclockwise starting from left
        self.left() | self.left().down() | self.down() | self.right().down()
            | self.right() | self.right().up() | self.up() | self.left().up()
    }

    pub fn jump_targets(self) -> Self {
        // counterclockwise starting from left.left
        self.left().left()
            | self.left().left().down()
            | self.left().left().down().down()
            | self.left().down().down()
            | self.down().down()
            | self.right().down().down()
            | self.right().right().down().down()
            | self.right().right().down()
            | self.right().right()
            | self.right().right().up()
            | self.right().right().up().up()
            | self.right().up().up()
            | self.up().up()
            | self.left().up().up()
            | self.left().left().up().up()
            | self.left().left().up()
    }

    pub fn map(self, sym: D4Symmetry) -> Tiles {
        let mut result = Tiles::empty();
        for c in self {
            result = result.set(c.map(sym))
        }
        result
    }
}

impl Display for Tiles {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        assert!(self.0 & Tiles::FULL_MASK == self.0);
        for y in (0..7).rev() {
            for x in 0..7 {
                let coord = Coord::from_xy(x, y);
                write!(f, "{}", if self.has(coord) { '1' } else { '.' })?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct MoveIterator<'a> {
    board: &'a AtaxxBoard,
}

#[derive(Debug)]
pub struct AllMoveIterator;

impl<'a> BoardAvailableMoves<'a, AtaxxBoard> for AtaxxBoard {
    type MoveIterator = MoveIterator<'a>;
    type AllMoveIterator = AllMoveIterator;

    fn all_possible_moves() -> Self::AllMoveIterator {
        AllMoveIterator
    }

    fn available_moves(&'a self) -> Self::MoveIterator {
        assert!(!self.is_done());
        MoveIterator { board: self }
    }
}

impl<'a> InternalIterator for AllMoveIterator {
    type Item = Move;

    fn find_map<R, F>(self, mut f: F) -> Option<R> where F: FnMut(Self::Item) -> Option<R> {
        if let Some(x) = f(Move::Pass) { return Some(x); };
        for to in Tiles::full() {
            if let Some(x) = f(Move::Copy { to }) { return Some(x); };
        }
        for from in Tiles::full() {
            for to in Tiles::coord(from).jump_targets() {
                if let Some(x) = f(Move::Jump { from, to }) { return Some(x); };
            }
        }
        None
    }
}

impl<'a> InternalIterator for MoveIterator<'a> {
    type Item = Move;

    fn find_map<R, F>(self, mut f: F) -> Option<R> where F: FnMut(Self::Item) -> Option<R> {
        let board = self.board;
        let next_tiles = board.tiles_pov().0;
        let free_tiles = board.free_tiles();

        // pass move, don't emit other moves afterwards
        if board.must_pass(next_tiles) {
            return f(Move::Pass);
        }

        // copy moves
        let copy_targets = free_tiles & next_tiles.copy_targets();
        for to in copy_targets {
            if let Some(x) = f(Move::Copy { to }) { return Some(x); }
        }

        // jump moves
        let jump_targets = free_tiles & next_tiles.jump_targets();
        for to in jump_targets {
            for from in next_tiles & Tiles::coord(to).jump_targets() {
                if let Some(x) = f(Move::Jump { from, to }) { return Some(x); }
            }
        }

        None
    }
}

impl Debug for Coord {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Coord({}, {})", self.x(), self.y())
    }
}

impl Coord {
    pub fn from_xy(x: u8, y: u8) -> Coord {
        assert!(x < 7);
        assert!(y < 7);
        Coord(y * 8 + x)
    }

    pub fn from_i(i: u8) -> Coord {
        Coord::from_xy(i % 8, i / 8)
    }

    pub fn x(self) -> u8 {
        self.0 % 8
    }

    pub fn y(self) -> u8 {
        self.0 / 8
    }

    pub fn i(self) -> u8 {
        self.0
    }

    pub fn distance(self, other: Coord) -> u8 {
        //TODO this can probably be written better, maybe even spacial case because w only care about 1, 2 and "other"
        let dx = abs_distance(self.x(), other.x());
        let dy = abs_distance(self.y(), other.y());
        max(dx, dy)
    }

    pub fn map(self, sym: D4Symmetry) -> Coord {
        let (x, y) = sym.map_xy(self.x(), self.y(), 6);
        Coord::from_xy(x, y)
    }
}

fn abs_distance(a: u8, b: u8) -> u8 {
    if a >= b { a - b } else { b - a }
}

impl Coord {
    pub fn to_uai(self) -> String {
        format!("{}{}", ('a' as u8 + self.x()) as char, self.y() + 1)
    }
}

impl Move {
    pub fn to_uai(self) -> String {
        match self {
            Move::Pass => "null".to_string(),
            Move::Copy { to } => to.to_uai(),
            Move::Jump { from, to } => format!("{}{}", from.to_uai(), to.to_uai())
        }
    }
}

fn player_symbol(player: Player) -> char {
    match player {
        Player::A => 'x',
        Player::B => 'o',
    }
}

impl AtaxxBoard {
    pub fn from_fen(fen: &str) -> AtaxxBoard {
        let mut board = AtaxxBoard::empty();

        let mut x = 0;
        let mut y = 6;
        let mut expect_next = false;
        let mut done = false;

        for c in fen.chars() {
            if done {
                assert!(c.is_ascii_digit() || c == ' ', "Unexpected '{}' near end in '{}'", c, fen);
                continue;
            }

            if expect_next {
                match c {
                    'x' => board.next_player = Player::A,
                    'o' => board.next_player = Player::B,
                    _ => panic!("Expected next player, got '{}' in '{}'", c, fen),
                }
                done = true;
                continue;
            }

            match c {
                'x' => board.tiles_a |= Tiles::coord(Coord::from_xy(x, y)),
                'o' => board.tiles_b |= Tiles::coord(Coord::from_xy(x, y)),
                '-' => board.gaps |= Tiles::coord(Coord::from_xy(x, y)),
                '/' => {
                    assert!(x == 7, "Row not yet complete, unexpected '/' in '{}'", fen);
                    x = 0;
                    y -= 1;
                    continue;
                }
                ' ' => {
                    assert!(x == 7 && y == 0, "Board not yet complete, unexpected ' ' in '{}'", fen);
                    expect_next = true;
                    continue;
                }
                d if d.is_ascii_digit() => {
                    let d = d.to_digit(10).unwrap() as u8;
                    assert!(0 < d && d <= 7, "Unexpected gap size {} in '{}'", d, fen);
                    x += d;
                    continue;
                }
                _ => panic!("Expected next tile or gap, got '{}' in '{}'", c, fen),
            }

            x = x.wrapping_add(1);
        }

        board.update_outcome();
        board
    }

    pub fn to_fen(&self) -> String {
        let mut s = String::new();

        for y in 0..7 {
            if y != 0 {
                write!(&mut s, "/").unwrap();
            }

            let mut space = 0;

            for x in 0..7 {
                let coord = Coord::from_xy(x, y);
                match self.tile(coord) {
                    None => {
                        if self.gaps.has(coord) {
                            write!(&mut s, "-").unwrap();
                        } else {
                            space += 1;
                        }
                    }
                    Some(player) => {
                        if space != 0 {
                            write!(&mut s, "{}", space).unwrap();
                            space = 0;
                        }
                        write!(&mut s, "{}", player_symbol(player)).unwrap();
                    }
                }
            }

            if space != 0 {
                write!(&mut s, "{}", space).unwrap();
            }
        }

        write!(&mut s, " {}", player_symbol(self.next_player)).unwrap();

        s
    }
}

impl Display for AtaxxBoard {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for y in (0..7).rev() {
            for x in 0..7 {
                let coord = Coord::from_xy(x, y);
                let tuple = (self.gaps.has(coord), self.tile(coord));
                let c = match tuple {
                    (true, None) => '-',
                    (false, None) => '.',
                    (false, Some(player)) => player_symbol(player),
                    (true, Some(_)) => unreachable!("Tile with block cannot have player"),
                };

                write!(f, "{}", c)?;
            }

            if y == 3 {
                write!(f, "    {}", player_symbol(self.next_player))?;
            }

            writeln!(f)?;
        }

        Ok(())
    }
}