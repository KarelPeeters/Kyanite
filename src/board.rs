use std::fmt::Debug;

use rand::Rng;
use rand::seq::IteratorRandom;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Player {
    A,
    B,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Outcome {
    WonBy(Player),
    Draw,
}

pub trait Board: Debug + Clone + Eq + PartialEq where for<'a> Self: BoardAvailableMoves<'a, Self> {
    type Move: Debug + Copy + Clone + Eq + PartialEq;

    /// Whether the player who plays a move can lose by playing that move.
    /// Symbolically whether `b.won_by() == Some(Winner::Player(b.next_player()))` can ever be true.
    fn can_lose_after_move() -> bool;

    /// Return minimum and maximum possible games lengths.
    fn game_length_bounds() -> (u32, Option<u32>);

    /// Return the next player to make a move.
    /// If the board is done this is the player that did not play the last move for consistency.
    fn next_player(&self) -> Player;

    /// Return whether the given move is available. Panics if this board is done.
    fn is_available_move(&self, mv: Self::Move) -> bool;

    /// Pick a random move from the `available_moves` with a uniform distribution. Panics if this board is done.
    /// Can be overridden for better performance.
    fn random_available_move(&self, rng: &mut impl Rng) -> Self::Move {
        self.available_moves().choose(rng).unwrap()
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
}

/// Trait to fake generic associated types, can be removed once that's stable.
/// See https://github.com/rust-lang/rust/issues/44265.
pub trait BoardAvailableMoves<'a, B: Board> {
    type MoveIterator: Iterator<Item=B::Move>;

    /// Return an iterator over available moves. Panics if this board is done.
    fn available_moves(&'a self) -> Self::MoveIterator;
}

impl Player {
    pub fn other(self) -> Player {
        match self {
            Player::A => Player::B,
            Player::B => Player::A,
        }
    }

    pub fn index(self) -> u32 {
        match self {
            Player::A => 0,
            Player::B => 1,
        }
    }

    pub fn sign(self, pov: Player) -> f32 {
        if self == pov { 1.0 } else { -1.0 }
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

    pub fn sign(self, pov: Player) -> f32 {
        match self {
            Outcome::WonBy(player) => player.sign(pov),
            Outcome::Draw => 0.0,
        }
    }
}

