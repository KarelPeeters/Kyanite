use crate::client::{Lichess, LichessResult};
use crate::models::board::{BoardState, Event};
use bytes::Bytes;
use futures_util::stream::Stream;
use serde_json::{from_value, Value};
use crate::models::chat::ChatMessage;

impl Lichess {
    pub async fn stream_incoming_events(
        &self,
    ) -> LichessResult<impl Stream<Item = LichessResult<Event>>> {
        let url = format!("{}/api/stream/event", self.base);
        let builder = self.client.get(&url);
        self.to_model_stream(builder).await
    }

    pub async fn create_a_seek(
        &self,
        time: u8,
        increment: u8,
        form_params: Option<&[(&str, &str)]>,
    ) -> LichessResult<impl Stream<Item = LichessResult<Bytes>>> {
        let url = format!("{}/api/board/seek", self.base);
        let mut form = vec![
            ("time", time.to_string()),
            ("increment", increment.to_string()),
        ];
        if let Some(params) = form_params {
            for (key, val) in params.iter() {
                form.push((key, val.to_string()))
            }
        }
        let builder = self.client.post(&url).form(&form);
        self.to_raw_bytes(builder).await
    }

    pub async fn stream_board_game_state(
        &self,
        game_id: &str,
    ) -> LichessResult<impl Stream<Item = LichessResult<BoardState>>> {
        let url = format!("{}/api/board/game/stream/{}", self.base, game_id);
        let builder = self.client.get(&url);
        self.to_model_stream(builder).await
    }

    pub async fn get_game_chat(&self, game_id: &str) -> LichessResult<Vec<ChatMessage>> {
        let url = format!("{}/api/board/game/{}/chat", self.base, game_id);
        let builder = self.client.get(url);
        self.to_model_full(builder).await
    }

    pub async fn make_a_board_move(
        &self,
        game_id: &str,
        r#move: &str,
        offering_draw: bool,
    ) -> LichessResult<()> {
        let url = format!("{}/api/board/game/{}/move/{}", self.base, game_id, r#move);
        let builder = self
            .client
            .post(&url)
            .query(&[("offeringDraw", offering_draw)]);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    pub async fn write_in_board_chat(
        &self,
        game_id: &str,
        room: &str,
        text: &str,
    ) -> LichessResult<()> {
        let url = format!("{}/api/board/game/{}/chat", self.base, game_id);
        let builder = self
            .client
            .post(&url)
            .form(&[("room", room), ("text", text)]);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    pub async fn abort_board_game(&self, game_id: &str) -> LichessResult<()> {
        let url = format!("{}/api/board/game/{}/abort", self.base, game_id);
        let builder = self.client.post(&url);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    pub async fn resign_board_game(&self, game_id: &str) -> LichessResult<()> {
        let url = format!("{}/api/board/game/{}/resign", self.base, game_id);
        let builder = self.client.post(&url);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    pub async fn handle_draw_offers(&self, game_id: &str, accept: bool) -> LichessResult<()> {
        let url = format!("{}/api/board/game/{}/draw/{}", self.base, game_id, accept);
        let builder = self.client.post(&url);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }
}
