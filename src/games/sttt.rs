use std::fmt;
use std::fmt::Debug;

use internal_iterator::InternalIterator;
use itertools::Itertools;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::board::{Board, BoardAvailableMoves, Outcome, Player};
use crate::util::bit_iter::BitIter;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Coord(u8);

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct STTTBoard {
    //TODO try u16 here, that makes Board a lot smaller and maybe even feasible to store in the tree?
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

    pub fn map_symmetry(&self, sym: Symmetry) -> STTTBoard {
        let mut grids = [0; 9];
        for oo in 0..9 {
            grids[sym.map_oo(oo) as usize] = sym.map_grid(self.grids[oo as usize])
        }

        STTTBoard {
            grids,
            main_grid: 0,
            last_move: self.last_move.map(|c| sym.map_coord(c)),
            next_player: self.next_player,
            outcome: self.outcome,
            macro_mask: sym.map_grid(self.macro_mask),
            macro_open: sym.map_grid(self.macro_open),
        }
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
                let os = get_nth_set_bit(!compact_grid(grid), index as u32);
                return Coord::from_oo(om as u8, os as u8);
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
}

impl<'a> BoardAvailableMoves<'a, STTTBoard> for STTTBoard {
    type MoveIterator = STTTMoveIterator<'a>;

    fn available_moves(&'a self) -> Self::MoveIterator {
        assert!(!self.is_done(), "Board must not be done");
        STTTMoveIterator { board: self }
    }
}

impl Coord {
    pub fn all() -> impl Iterator<Item=Coord> {
        (0..81).map(|o| Self::from_o(o))
    }

    pub fn all_yx() -> impl Iterator<Item=Coord> {
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
pub struct STTTMoveIterator<'a> {
    board: &'a STTTBoard,
}

impl<'a> InternalIterator for STTTMoveIterator<'a> {
    type Item = Coord;

    fn find_map<R, F>(self, mut f: F) -> Option<R> where F: FnMut(Self::Item) -> Option<R> {
        for om in BitIter::new(self.board.macro_mask) {
            let free_grid = (!compact_grid(self.board.grids[om as usize])) & STTTBoard::FULL_MASK;
            for os in BitIter::new(free_grid) {
                if let Some(r) = f(Coord::from_oo(om as u8, os as u8)) {
                    return Some(r);
                }
            }
        }

        None
    }
}

/// A symmetry group element for Board transformations. Can represent any combination of
/// flips, rotating and transposing, which result in 8 distinct elements.
///
/// The `Default::default()` value means no transformation.
///
/// The internal representation is such that first x and y are transposed,
/// then each axis is flipped separately.
#[derive(Debug, Copy, Clone)]
pub struct Symmetry {
    pub transpose: bool,
    pub flip_x: bool,
    pub flip_y: bool,
}

impl Default for Symmetry {
    fn default() -> Self {
        Symmetry { transpose: false, flip_x: false, flip_y: false }
    }
}

impl Distribution<Symmetry> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Symmetry {
        Symmetry { transpose: rng.gen(), flip_x: rng.gen(), flip_y: rng.gen() }
    }
}

impl Symmetry {
    pub fn all() -> impl Iterator<Item=Symmetry> {
        (0..8).map(|i| Symmetry {
            transpose: i & 0b100 != 0,
            flip_x: i & 0b010 != 0,
            flip_y: i & 0b001 != 0,
        })
    }

    pub fn inverse(self) -> Symmetry {
        Symmetry {
            transpose: self.transpose,
            flip_x: if self.transpose { self.flip_y } else { self.flip_x },
            flip_y: if self.transpose { self.flip_x } else { self.flip_y },
        }
    }

    pub fn map_coord(self, coord: Coord) -> Coord {
        Coord::from_oo(self.map_oo(coord.om()), self.map_oo(coord.os()))
    }

    pub fn map_oo(self, oo: u8) -> u8 {
        let (mut x, mut y) = (oo % 3, oo / 3);
        if self.transpose { std::mem::swap(&mut x, &mut y) };
        if self.flip_x { x = 2 - x };
        if self.flip_y { y = 2 - y };
        x + y * 3
    }

    fn map_grid(self, grid: u32) -> u32 {
        let mut result = 0;
        for oo_input in 0..9 {
            let oo_result = self.map_oo(oo_input);
            let get = (grid >> oo_input) & 0b1_000_000_001;
            result |= get << oo_result;
        }
        result
    }
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

fn get_nth_set_bit(mut x: u32, n: u32) -> u32 {
    for _ in 0..n {
        x &= x.wrapping_sub(1);
    }
    x.trailing_zeros()
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

#[cfg(test)]
mod test {
    use internal_iterator::InternalIterator;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand::seq::SliceRandom;

    use crate::board::{Board, BoardAvailableMoves};
    use crate::games::sttt::{Coord, STTTBoard, Symmetry};
    use crate::util::board_gen::random_board_with_moves;

    #[test]
    //TODO move this test somewhere more generic where it can be run for all games, not just sttt
    fn test_random_distribution() {
        let mut board = STTTBoard::default();
        let mut rand = SmallRng::seed_from_u64(0);

        while !board.is_done() {
            let moves: Vec<Coord> = board.available_moves().collect();

            let mut counts: [i32; 81] = [0; 81];
            for _ in 0..1_000_000 {
                counts[board.random_available_move(&mut rand).o() as usize] += 1;
            }

            let avg = (1_000_000 / moves.len()) as i32;

            for (mv, &count) in counts.iter().enumerate() {
                if moves.contains(&Coord::from_o(mv as u8)) {
                    debug_assert!((count.wrapping_sub(avg)).abs() < 10_000, "uniformly distributed")
                } else {
                    assert_eq!(count, 0, "only actual moves returned")
                }
            }

            let mv = moves.choose(&mut rand).unwrap().o();
            board.play(Coord::from_o(mv as u8));
        }
    }

    #[test]
    fn symmetries() {
        let mut rng = SmallRng::seed_from_u64(5);
        let board = random_board_with_moves(&STTTBoard::default(), 10, &mut rng);
        println!("Original:\n{}", board);

        for i in 0..8 {
            let sym = Symmetry {
                transpose: i & 0b001 != 0,
                flip_x: i & 0b010 != 0,
                flip_y: i & 0b100 != 0,
            };
            let sym_inv = sym.inverse();

            println!("{:?}", sym);
            println!("inverse: {:?}", sym_inv);

            let mapped = board.map_symmetry(sym);
            let back = mapped.map_symmetry(sym_inv);

            // these prints test that the board is consistent enough to print it
            println!("Mapped:\n{}", mapped);
            println!("Back:\n{}", back);

            if i == 0 {
                assert_eq!(board, mapped);
            }
            assert_eq!(board, back);

            let mut expected_moves: Vec<Coord> = board.available_moves().map(|c| sym.map_coord(c)).collect();
            let mut actual_moves: Vec<Coord> = mapped.available_moves().collect();

            expected_moves.sort_by_key(|c| c.o());
            actual_moves.sort_by_key(|c| c.o());

            assert_eq!(expected_moves, actual_moves);
        }
    }
}