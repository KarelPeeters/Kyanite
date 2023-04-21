use board_game::board::Board;
use std::fmt::Debug;

use async_trait::async_trait;
use board_game::ai::Bot;

use board_game::board::BoardDone;

#[async_trait]
pub trait AsyncBot<B: Board> {
    async fn select_move(&mut self, board: &B) -> Result<B::Move, BoardDone>;
}

#[derive(Debug)]
pub struct WrapAsync<T>(pub T);

#[async_trait]
impl<B: Board, T: Bot<B> + Send> AsyncBot<B> for WrapAsync<T> {
    async fn select_move(&mut self, board: &B) -> Result<B::Move, BoardDone> {
        self.0.select_move(board)
    }
}
