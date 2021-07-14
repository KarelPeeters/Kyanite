use std::fmt::{Debug, Display};
use std::hash::Hash;

use internal_iterator::InternalIterator;
use rand::Rng;

use crate::symmetry::Symmetry;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Player {
    A,
    B,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Outcome {
    WonBy(Player),
    Draw,
}

/// The representation of a game state.
pub trait Board: Debug + Display + Clone + Eq + Hash + Send + Sync where for<'a> Self: BoardAvailableMoves<'a, Self> {
    type Move: Debug + Copy + Eq + Ord + Hash + Send + Sync;
    type Symmetry: Symmetry;

    /// Whether the player who plays a move can lose by playing that move.
    /// Symbolically whether `b.won_by() == Some(Winner::Player(b.next_player()))` can ever be true.
    /// This may be pessimistic, returning `true` is always correct.
    fn can_lose_after_move() -> bool;

    /// Return minimum and maximum possible games lengths. These bounds may be pessimistic,
    /// returning `(0, None)` is always correct.
    fn game_length_bounds() -> (u32, Option<u32>);

    /// Return the next player to make a move.
    /// If the board is done this is the player that did not play the last move for consistency.
    fn next_player(&self) -> Player;

    /// Return whether the given move is available. Panics if this board is done.
    fn is_available_move(&self, mv: Self::Move) -> bool;

    /// Pick a random move from the `available_moves` with a uniform distribution. Panics if this board is done.
    /// Can be overridden for better performance.
    fn random_available_move(&self, rng: &mut impl Rng) -> Self::Move {
        let count = self.available_moves().count();
        let index = rng.gen_range(0..count);
        self.available_moves().nth(index).unwrap()
    }

    /// Play the move `mv`, modifying this board.
    /// Panics if this board is done or if the move is not available or valid for this board.
    fn play(&mut self, mv: Self::Move);

    /// Clone this board, play `mv` on it and return the new board.
    fn clone_and_play(&self, mv: Self::Move) -> Self {
        let mut next = self.clone();
        next.play(mv);
        next
    }

    /// The outcome of this board, is `None` when this games is not done yet.
    fn outcome(&self) -> Option<Outcome>;

    /// Whether this games is done.
    fn is_done(&self) -> bool {
        self.outcome().is_some()
    }

    /// Map this board under the given symmetry.
    fn map(&self, sym: Self::Symmetry) -> Self;

    /// Map a move under the given symmetry.
    fn map_move(sym: Self::Symmetry, mv: Self::Move) -> Self::Move;
}

/// Trait to fake generic associated types, can be removed once that's stable.
/// See https://github.com/rust-lang/rust/issues/44265.
pub trait BoardAvailableMoves<'a, B: Board> {
    type MoveIterator: InternalIterator<Item=B::Move>;
    type AllMoveIterator: InternalIterator<Item=B::Move>;

    /// All theoretically possible moves, for any possible board.
    /// Moves returned by `available_moves` will always be a subset of these moves.
    /// The ordering must be consistent with `available_moves`.
    fn all_possible_moves() -> Self::AllMoveIterator;

    /// Return an iterator over available moves, is always nonempty. No guarantees are made about the ordering except
    /// that it stays consistent when the board is not modified.
    /// Panics if this board is done.
    fn available_moves(&'a self) -> Self::MoveIterator;
}

impl Player {
    pub fn other(self) -> Player {
        match self {
            Player::A => Player::B,
            Player::B => Player::A,
        }
    }

    pub fn index(self) -> u8 {
        match self {
            Player::A => 0,
            Player::B => 1,
        }
    }

    pub fn sign(self, pov: Player) -> i8 {
        if self == pov { 1 } else { -1 }
    }
}

impl Outcome {
    pub fn other(self) -> Outcome {
        match self {
            Outcome::WonBy(player) => Outcome::WonBy(player.other()),
            Outcome::Draw => Outcome::Draw,
        }
    }

    pub fn unwrap_player(self) -> Player {
        match self {
            Outcome::WonBy(player) => player,
            Outcome::Draw => panic!("Expected a player, got {:?}", self),
        }
    }

    pub fn sign(self, pov: Player) -> i8 {
        match self {
            Outcome::WonBy(player) => player.sign(pov),
            Outcome::Draw => 0,
        }
    }

    pub fn inf_sign(self, pov: Player) -> f32 {
        match self {
            Outcome::WonBy(player) => player.sign(pov) as f32 * f32::INFINITY,
            Outcome::Draw => 0.0,
        }
    }
}

