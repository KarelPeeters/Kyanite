use crate::client::{Lichess, LichessResult};
use crate::models::board::BoardState;
use futures_util::stream::Stream;
use serde_json::{from_value, Value};

impl Lichess {
    pub async fn upgrade_to_bot_account(&self) -> LichessResult<()> {
        let url = format!("{}/api/bot/account/upgrade", self.base);
        let builder = self.client.post(&url);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    pub async fn stream_bot_game_state(
        &self,
        game_id: &str,
    ) -> LichessResult<impl Stream<Item = LichessResult<BoardState>>> {
        let url = format!("{}/api/bot/game/stream/{}", self.base, game_id);
        let builder = self.client.get(&url);
        self.to_model_stream(builder).await
    }

    pub async fn make_a_bot_move(
        &self,
        game_id: &str,
        r#move: &str,
        offering_draw: bool,
    ) -> LichessResult<()> {
        let url = format!("{}/api/bot/game/{}/move/{}", self.base, game_id, r#move);
        let builder = self
            .client
            .post(&url)
            .query(&[("offeringDraw", offering_draw)]);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    pub async fn write_in_bot_chat(
        &self,
        game_id: &str,
        room: &str,
        text: &str,
    ) -> LichessResult<()> {
        let url = format!("{}/api/bot/game/{}/chat", self.base, game_id);
        let builder = self
            .client
            .post(&url)
            .form(&[("room", room), ("text", text)]);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    pub async fn abort_bot_game(&self, game_id: &str) -> LichessResult<()> {
        let url = format!("{}/api/bot/game/{}/abort", self.base, game_id);
        let builder = self.client.post(&url);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    pub async fn resign_bot_game(&self, game_id: &str) -> LichessResult<()> {
        let url = format!("{}/api/bot/game/{}/resign", self.base, game_id);
        let builder = self.client.post(&url);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    // pub async fn handle_draw_offers(&self, game_id: &str, accept: bool) -> LichessResult<()> {
    //     let url = format!("{}/api/bot/game/{}/draw/{}", self.base, game_id, accept);
    //     let builder = self.client.post(&url);
    //     let ok_json = self.to_model_full::<Value>(builder);
    //     assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
    //     Ok(())
    // }
}
