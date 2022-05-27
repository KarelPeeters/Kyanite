use board_game::board::Board;
use std::fmt::Debug;

use async_trait::async_trait;
use board_game::ai::Bot;

#[async_trait]
pub trait AsyncBot<B: Board>: Debug {
    async fn select_move(&mut self, board: &B) -> B::Move;
}

// TODO is this implementation a good idea? Bots are written to consume as much (blocking) CPU as they want
#[async_trait]
impl<B: Board, T: Bot<B> + Send> AsyncBot<B> for T {
    async fn select_move(&mut self, board: &B) -> B::Move {
        Bot::select_move(self, board)
    }
}
