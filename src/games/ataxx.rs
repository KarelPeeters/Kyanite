use std::cmp::max;
use std::fmt::{Debug, Display, Formatter};

use internal_iterator::InternalIterator;
use newtype_ops::newtype_ops;

use crate::board::{Board, BoardAvailableMoves, Outcome, Player};
use crate::util::bit_iter::BitIter;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Coord(u8);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
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

    pub fn block(&self, coord: Coord) -> bool {
        self.gaps.has(coord)
    }

    pub fn free_tiles(&self) -> Tiles {
        !(self.tiles_a | self.tiles_b | self.gaps)
    }

    pub fn must_pass(&self, tiles: Tiles) -> bool {
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
}

impl Board for AtaxxBoard {
    type Move = Move;

    fn can_lose_after_move() -> bool {
        false
    }

    fn game_length_bounds() -> (u32, Option<u32>) {
        (0, None)
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

        //check if any player can make a move and set outcome if not
        if self.must_pass(self.tiles_a) && self.must_pass(self.tiles_b) {
            let count_a = self.tiles_a.count();
            let count_b = self.tiles_b.count();

            let outcome = if count_a > count_b {
                Outcome::WonBy(Player::A)
            } else if count_a < count_b {
                Outcome::WonBy(Player::B)
            } else {
                Outcome::Draw
            };
            self.outcome = Some(outcome)
        }

        self.next_player = self.next_player.other();
    }

    fn outcome(&self) -> Option<Outcome> {
        self.outcome
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

impl Tiles {
    pub const FULL_MASK: u64 = 0x7F_7F_7F_7F_7F_7F_7F;

    pub const CORNERS_A: Tiles = Tiles(0x_40_00_00_00_00_00_01);
    pub const CORNERS_B: Tiles = Tiles(0x_01_00_00_00_00_00_40);

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

    pub fn up(self) -> Self {
        Tiles((self.0 >> 8) & Self::FULL_MASK)
    }

    pub fn down(self) -> Self {
        Tiles((self.0 << 8) & Self::FULL_MASK)
    }

    pub fn copy_targets(self) -> Self {
        //clockwise starting from left
        self.left() | self.left().up() | self.up() | self.right().up()
            | self.right() | self.right().down() | self.down() | self.left().down()
    }

    pub fn jump_targets(self) -> Self {
        //clockwise starting from left.left
        self.left().left()
            | self.left().left().up()
            | self.left().left().up().up()
            | self.left().up().up()
            | self.up().up()
            | self.right().up().up()
            | self.right().right().up().up()
            | self.right().right().up()
            | self.right().right()
            | self.right().right().down()
            | self.right().right().down().down()
            | self.right().down().down()
            | self.down().down()
            | self.left().down().down()
            | self.left().left().down().down()
            | self.left().left().down()
    }

    pub fn iter(self) -> impl Iterator<Item=Coord> {
        BitIter::new(self.0).map(|i| Coord::from_i(i as u8))
    }
}

impl Display for Tiles {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        assert!(self.0 & Tiles::FULL_MASK == self.0);
        for y in 0..7 {
            for x in 0..7 {
                let coord = Coord::from_xy(x, y);
                write!(f, "{}", if self.has(coord) { '1' } else { '.' })?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

pub struct MoveIterator<'a> {
    board: &'a AtaxxBoard,
}

impl<'a> BoardAvailableMoves<'a, AtaxxBoard> for AtaxxBoard {
    type MoveIterator = MoveIterator<'a>;

    fn available_moves(&'a self) -> Self::MoveIterator {
        assert!(!self.is_done());
        MoveIterator { board: self }
    }
}

impl<'a> InternalIterator for MoveIterator<'a> {
    type Item = Move;

    fn find_map<R, F>(self, mut f: F) -> Option<R> where F: FnMut(Self::Item) -> Option<R> {
        let board = self.board;
        let next_tiles = board.tiles_pov().0;
        let free_tiles = board.free_tiles();

        // pass move
        if board.must_pass(next_tiles) {
            return f(Move::Pass);
        }

        // copy moves
        for to in (free_tiles & next_tiles.copy_targets()).iter() {
            if let Some(x) = f(Move::Copy { to }) { return Some(x); }
        }

        // jump moves
        for from in next_tiles.iter() {
            for to in (free_tiles & Tiles::coord(from).jump_targets()).iter() {
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
}

fn abs_distance(a: u8, b: u8) -> u8 {
    if a >= b { a - b } else { b - a }
}

impl Display for AtaxxBoard {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for y in 0..7 {
            for x in 0..7 {
                let coord = Coord::from_xy(x, y);
                let tuple = (self.block(coord), self.tile(coord));
                let c = match tuple {
                    (true, None) => 'X',
                    (false, None) => '.',
                    (false, Some(Player::A)) => 'a',
                    (false, Some(Player::B)) => 'b',
                    (true, Some(_)) => unreachable!("Tile with block cannot have player"),
                };

                write!(f, "{}", c)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}