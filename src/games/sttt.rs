use std::fmt;
use std::fmt::Debug;

use internal_iterator::{InternalIterator, Internal, IteratorExt};
use itertools::Itertools;
use rand::Rng;

use crate::board::{Board, BoardAvailableMoves, Outcome, Player};
use crate::symmetry::D4Symmetry;
use crate::util::bits::{BitIter, get_nth_set_bit};

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Coord(u8);

#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub struct STTTBoard {
    grids: [u32; 9],
    main_grid: u32,

    last_move: Option<Coord>,
    next_player: Player,
    outcome: Option<Outcome>,

    macro_mask: u32,
    macro_open: u32,
}

impl Default for STTTBoard {
    fn default() -> STTTBoard {
        STTTBoard {
            grids: [0; 9],
            main_grid: 0,
            last_move: None,
            next_player: Player::A,
            outcome: None,
            macro_mask: STTTBoard::FULL_MASK,
            macro_open: STTTBoard::FULL_MASK,
        }
    }
}

impl STTTBoard {
    const FULL_MASK: u32 = 0b111_111_111;

    pub fn tile(&self, coord: Coord) -> Option<Player> {
        get_player(self.grids[coord.om() as usize], coord.os())
    }

    pub fn macr(&self, om: u8) -> Option<Player> {
        debug_assert!(om < 9);
        get_player(self.main_grid, om)
    }

    pub fn is_macro_open(&self, om: u8) -> bool {
        debug_assert!(om < 9);
        has_bit(self.macro_open, om)
    }

    /// Return the number of non-empty tiles.
    pub fn count_tiles(&self) -> u32 {
        self.grids.iter().map(|tile| tile.count_ones()).sum()
    }

    fn set_tile_and_update(&mut self, player: Player, coord: Coord) {
        let om = coord.om();
        let os = coord.os();
        let p = (9 * player.index()) as u8;

        //set tile and macro, check win
        let new_grid = self.grids[om as usize] | (1 << (os + p));
        self.grids[om as usize] = new_grid;

        let grid_win = is_win_grid((new_grid >> p) & STTTBoard::FULL_MASK);
        if grid_win {
            let new_main_grid = self.main_grid | (1 << (om + p));
            self.main_grid = new_main_grid;

            if is_win_grid((new_main_grid >> p) & STTTBoard::FULL_MASK) {
                self.outcome = Some(Outcome::WonBy(player));
            }
        }

        //update macro masks, remove bit from open and recalculate mask
        if grid_win || new_grid.count_ones() == 9 {
            self.macro_open &= !(1 << om);
            if self.macro_open == 0 && self.outcome.is_none() {
                self.outcome = Some(Outcome::Draw);
            }
        }
        self.macro_mask = self.calc_macro_mask(os);
    }

    fn calc_macro_mask(&self, os: u8) -> u32 {
        if has_bit(self.macro_open, os) {
            1u32 << os
        } else {
            self.macro_open
        }
    }
}

impl Board for STTTBoard {
    type Move = Coord;
    type Symmetry = D4Symmetry;

    fn can_lose_after_move() -> bool {
        false
    }

    fn game_length_bounds() -> (u32, Option<u32>) {
        (9 + 8, Some(81))
    }

    fn next_player(&self) -> Player {
        self.next_player
    }

    fn is_available_move(&self, mv: Self::Move) -> bool {
        assert!(!self.is_done(), "Board must not be done");

        has_bit(self.macro_mask, mv.om()) &&
            !has_bit(compact_grid(self.grids[mv.om() as usize]), mv.os())
    }

    fn random_available_move(&self, rng: &mut impl Rng) -> Self::Move {
        // TODO we can also implement size_hint and skip for the available move iterator,
        //   then we don't need this complicated body any more

        assert!(!self.is_done(), "Board must not be done");

        let mut count = 0;
        for om in BitIter::new(self.macro_mask) {
            count += 9 - self.grids[om as usize].count_ones();
        }

        let mut index = rng.gen_range(0..count);

        for om in BitIter::new(self.macro_mask) {
            let grid = self.grids[om as usize];
            let grid_count = 9 - grid.count_ones();

            if index < grid_count {
                let os = get_nth_set_bit(!compact_grid(grid), index);
                return Coord::from_oo(om, os);
            }

            index -= grid_count;
        }

        unreachable!()
    }

    fn play(&mut self, mv: Self::Move) {
        assert!(!self.is_done(), "Board must not be done");
        assert!(self.is_available_move(mv), "move not available");

        //do actual move
        self.set_tile_and_update(self.next_player, mv);

        //update for next player
        self.last_move = Some(mv);
        self.next_player = self.next_player.other()
    }

    fn outcome(&self) -> Option<Outcome> {
        self.outcome
    }

    fn map(&self, sym: D4Symmetry) -> STTTBoard {
        let mut grids = [0; 9];
        for oo in 0..9 {
            grids[map_oo(sym, oo) as usize] = map_grid(sym, self.grids[oo as usize])
        }

        STTTBoard {
            grids,
            main_grid: 0,
            last_move: self.last_move.map(|c| Self::map_move(sym, c)),
            next_player: self.next_player,
            outcome: self.outcome,
            macro_mask: map_grid(sym, self.macro_mask),
            macro_open: map_grid(sym, self.macro_open),
        }
    }

    fn map_move(sym: D4Symmetry, mv: Coord) -> Coord {
        Coord::from_oo(map_oo(sym, mv.om()), map_oo(sym, mv.os()))
    }
}

impl<'a> BoardAvailableMoves<'a, STTTBoard> for STTTBoard {
    type MoveIterator = STTTMoveIterator<'a>;
    type AllMoveIterator = Internal<CoordIter>;

    fn all_possible_moves() -> Self::AllMoveIterator {
        Coord::all().into_internal()
    }

    fn available_moves(&'a self) -> Self::MoveIterator {
        assert!(!self.is_done(), "Board must not be done");
        STTTMoveIterator { board: self }
    }
}

pub type CoordIter = std::iter::Map<std::ops::Range<u8>, fn(u8) -> Coord>;

impl Coord {

    pub fn all() -> CoordIter {
        (0..81).map(|o| Self::from_o(o))
    }

    pub fn all_yx() -> CoordIter {
        (0..81).map(|i| Self::from_xy(i % 9, i / 9))
    }

    pub fn from_oo(om: u8, os: u8) -> Coord {
        debug_assert!(om < 9);
        debug_assert!(os < 9);
        Coord(9 * om + os)
    }

    pub fn from_o(o: u8) -> Coord {
        debug_assert!(o < 81);
        Coord(o)
    }

    pub fn from_xy(x: u8, y: u8) -> Coord {
        debug_assert!(x < 9 && y < 9);
        Coord(((x / 3) + (y / 3) * 3) * 9 + ((x % 3) + (y % 3) * 3))
    }

    pub fn om(self) -> u8 {
        self.0 / 9
    }

    pub fn os(self) -> u8 {
        self.0 % 9
    }

    pub fn o(self) -> u8 {
        9 * self.om() + self.os()
    }

    pub fn yx(self) -> u8 {
        9 * self.y() + self.x()
    }

    pub fn x(self) -> u8 {
        (self.om() % 3) * 3 + (self.os() % 3)
    }

    pub fn y(self) -> u8 {
        (self.om() / 3) * 3 + (self.os() / 3)
    }
}

impl Debug for Coord {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "({}, {})", self.om(), self.os())
    }
}

//TODO implement a size hint
//TODO look into other iterator speedup functions that can be implemented
#[derive(Debug)]
pub struct STTTMoveIterator<'a> {
    board: &'a STTTBoard,
}

impl<'a> InternalIterator for STTTMoveIterator<'a> {
    type Item = Coord;

    fn find_map<R, F>(self, mut f: F) -> Option<R> where F: FnMut(Self::Item) -> Option<R> {
        for om in BitIter::new(self.board.macro_mask) {
            let free_grid = (!compact_grid(self.board.grids[om as usize])) & STTTBoard::FULL_MASK;
            for os in BitIter::new(free_grid) {
                if let Some(r) = f(Coord::from_oo(om, os)) {
                    return Some(r);
                }
            }
        }

        None
    }
}

fn map_oo(sym: D4Symmetry, oo: u8) -> u8 {
    let (x, y) = sym.map_xy(oo % 3, oo / 3, 2);
    x + y * 3
}

fn map_grid(sym: D4Symmetry, grid: u32) -> u32 {
    // this could be implemented faster but it's not on a hot path
    let mut result = 0;
    for oo_input in 0..9 {
        let oo_result = map_oo(sym, oo_input);
        let get = (grid >> oo_input) & 0b1_000_000_001;
        result |= get << oo_result;
    }
    result
}

fn is_win_grid(grid: u32) -> bool {
    debug_assert!(has_mask(STTTBoard::FULL_MASK, grid));

    const WIN_GRIDS: [u32; 16] = [
        2155905152, 4286611584, 4210076288, 4293962368,
        3435954304, 4291592320, 4277971584, 4294748800,
        2863300736, 4294635760, 4210731648, 4294638320,
        4008607872, 4294897904, 4294967295, 4294967295
    ];
    has_bit(WIN_GRIDS[(grid / 32) as usize], (grid % 32) as u8)
}

fn has_bit(x: u32, i: u8) -> bool {
    ((x >> i) & 1) != 0
}

fn has_mask(x: u32, mask: u32) -> bool {
    x & mask == mask
}

fn compact_grid(grid: u32) -> u32 {
    (grid | grid >> 9) & STTTBoard::FULL_MASK
}

fn get_player(grid: u32, index: u8) -> Option<Player> {
    if has_bit(grid, index) {
        Some(Player::A)
    } else if has_bit(grid, index + 9) {
        Some(Player::B)
    } else {
        None
    }
}

fn symbol_from_tile(board: &STTTBoard, coord: Coord) -> char {
    let is_last = Some(coord) == board.last_move;
    let is_available = board.is_available_move(coord);
    let player = board.tile(coord);
    symbol_from_tuple(is_available, is_last, player)
}

fn symbol_from_tuple(is_available: bool, is_last: bool, player: Option<Player>) -> char {
    let tuple = (is_available, is_last, player);
    match tuple {
        (false, false, Some(Player::A)) => 'x',
        (false, true, Some(Player::A)) => 'X',
        (false, false, Some(Player::B)) => 'o',
        (false, true, Some(Player::B)) => 'O',
        (true, false, None) => '.',
        (false, false, None) => ' ',
        _ => unreachable!("Invalid tile state {:?}", tuple)
    }
}

fn symbol_to_tuple(c: char) -> (bool, bool, Option<Player>) {
    match c {
        'x' => (false, false, Some(Player::A)),
        'X' => (false, true, Some(Player::A)),
        'o' => (false, false, Some(Player::B)),
        'O' => (false, true, Some(Player::B)),
        ' ' => (false, false, None),
        '.' => (true, false, None),
        _ => panic!("unexpected character '{}'", c)
    }
}

impl fmt::Display for STTTBoard {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for y in 0..9 {
            if y == 3 || y == 6 {
                writeln!(f, "---+---+---     +---+")?;
            }

            for x in 0..9 {
                if x == 3 || x == 6 {
                    write!(f, "|")?;
                }
                write!(f, "{}", symbol_from_tile(self, Coord::from_xy(x, y)))?;
            }

            if (3..6).contains(&y) {
                write!(f, "     |")?;
                let ym = y - 3;
                for xm in 0..3 {
                    let om = xm + 3 * ym;
                    write!(f, "{}", symbol_from_tuple(self.is_macro_open(om), false, self.macr(om)))?;
                }
                write!(f, "|")?;
            }

            writeln!(f)?;
        }

        Ok(())
    }
}

pub fn board_to_compact_string(board: &STTTBoard) -> String {
    Coord::all().map(|coord| symbol_from_tile(board, coord)).join("")
}

pub fn board_from_compact_string(s: &str) -> STTTBoard {
    assert!(s.chars().count() == 81, "compact string should have length 81");

    let mut board = STTTBoard::default();
    let mut last_move = None;

    for (o, c) in s.chars().enumerate() {
        let coord = Coord::from_o(o as u8);
        let (_, last, player) = symbol_to_tuple(c);

        if last {
            assert!(last_move.is_none(), "Compact string cannot contain multiple last moves");
            let player = player.expect("Last move must have been played by a player");
            last_move = Some((player, coord));
        }

        if let Some(player) = player {
            board.set_tile_and_update(player, coord);
        }
    }

    if let Some((last_player, last_coord)) = last_move {
        board.set_tile_and_update(last_player, last_coord);
        board.last_move = Some(last_coord);
        board.next_player = last_player.other()
    }

    board
}
