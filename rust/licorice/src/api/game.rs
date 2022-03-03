use crate::client::{Lichess, LichessResult};
use crate::models::game::{Game, TVChannels, UserGame};
use bytes::Bytes;
use futures_util::stream::Stream;
use serde_json::{from_value, Value};

impl Lichess {
    pub async fn export_one_game_pgn(
        &self,
        game_id: &str,
        query_params: Option<&Vec<(&str, &str)>>,
    ) -> LichessResult<String> {
        let url = format!("{}/game/export/{}", self.base, game_id);
        let mut builder = self
            .client
            .get(&url)
            .header("Accept", "application/x-chess-pgn");
        if let Some(query) = query_params {
            builder = builder.query(&query)
        }
        self.to_raw_str(builder).await
    }

    pub async fn export_one_game_json(
        &self,
        game_id: &str,
        query_params: Option<&Vec<(&str, &str)>>,
    ) -> LichessResult<Game> {
        let url = format!("{}/game/export/{}", self.base, game_id);
        let mut builder = self.client.get(&url).header("Accept", "application/json");
        if let Some(query) = query_params {
            builder = builder.query(&query)
        }
        self.to_model_full(builder).await
    }

    pub async fn export_ongoing_game_pgn(
        &self,
        username: &str,
        query_params: Option<&Vec<(&str, &str)>>,
    ) -> LichessResult<String> {
        let url = format!("{}/api/user/{}/current-game", self.base, username);
        let mut builder = self
            .client
            .get(&url)
            .header("Accept", "application/x-chess-pgn");
        if let Some(query) = query_params {
            builder = builder.query(query)
        }
        self.to_raw_str(builder).await
    }

    pub async fn export_ongoing_game_json(
        &self,
        username: &str,
        query_params: Option<&Vec<(&str, &str)>>,
    ) -> LichessResult<Game> {
        let url = format!("{}/api/user/{}/current-game", self.base, username);
        let mut builder = self.client.get(&url).header("Accept", "application/json");
        if let Some(query) = query_params {
            builder = builder.query(query)
        }
        self.to_model_full(builder).await
    }

    pub async fn export_all_games_pgn(
        &self,
        username: &str,
        query_params: Option<&Vec<(&str, &str)>>,
    ) -> LichessResult<impl Stream<Item = LichessResult<Bytes>>> {
        let url = format!("{}/api/games/user/{}", self.base, username);
        let mut builder = self
            .client
            .get(&url)
            .header("Accept", "application/x-chess-pgn");
        if let Some(query) = query_params {
            builder = builder.query(query)
        }
        self.to_raw_bytes(builder).await
    }

    pub async fn export_all_games_json(
        &self,
        username: &str,
        query_params: Option<&Vec<(&str, &str)>>,
    ) -> LichessResult<impl Stream<Item = LichessResult<Game>>> {
        let url = format!("{}/api/games/user/{}", self.base, username);
        let mut builder = self
            .client
            .get(&url)
            .header("Accept", "application/x-ndjson");
        if let Some(query) = query_params {
            builder = builder.query(query)
        }
        self.to_model_stream(builder).await
    }

    pub async fn export_games_by_ids_json(
        &self,
        ids: &[&str],
        query_params: Option<&Vec<(&str, &str)>>,
    ) -> LichessResult<impl Stream<Item = LichessResult<Game>>> {
        let url = format!("{}/games/export/_ids", self.base);
        let mut builder = self
            .client
            .post(&url)
            .body(ids.join(","))
            .header("Accept", "application/x-ndjson");
        if let Some(query) = query_params {
            builder = builder.query(query)
        }
        self.to_model_stream(builder).await
    }

    pub async fn stream_current_games(
        &self,
        ids: &[&str],
    ) -> LichessResult<impl Stream<Item = LichessResult<Game>>> {
        let url = format!("{}/api/stream/games-by-users", self.base);
        let builder = self.client.post(&url).body(ids.join(","));
        self.to_model_stream(builder).await
    }

    pub async fn get_ongoing_games(&self, nb_games: u8) -> LichessResult<Vec<UserGame>> {
        let url = format!("{}/api/account/playing", self.base);
        let builder = self.client.get(&url).query(&[("nb", nb_games)]);
        let now_playing_json = self.to_model_full::<Value>(builder);
        from_value(now_playing_json.await?["nowPlaying"].take()).map_err(Into::into)
    }

    pub async fn get_current_tv_games(&self) -> LichessResult<TVChannels> {
        let url = format!("{}/tv/channels", self.base);
        let builder = self.client.get(&url);
        self.to_model_full(builder).await
    }

    pub async fn import_one_game(&self, pgn: &str) -> LichessResult<String> {
        let url = format!("{}/api/import", self.base);
        let builder = self.client.post(&url).form(&[("pgn", pgn)]);
        let url_json = self.to_model_full::<Value>(builder);
        from_value(url_json.await?["url"].take()).map_err(Into::into)
    }
}
