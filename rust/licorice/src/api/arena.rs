use crate::client::{Lichess, LichessResult};
use crate::models::{
    game::Game,
    tournament::{ArenaSchedule, ArenaTournament, PlayerStandings, TeamStandings},
};
use bytes::Bytes;
use futures_util::stream::Stream;

impl Lichess {
    pub async fn arena_current(&self) -> LichessResult<ArenaSchedule> {
        let url = format!("{}/api/tournament", self.base);
        let builder = self.client.get(&url);
        self.to_model_full(builder).await
    }

    pub async fn arena_new(
        &self,
        clock_time: u16, // seconds
        clock_increment: u8,
        minutes: u16,
        form_params: Option<&[(&str, &str)]>,
    ) -> LichessResult<ArenaTournament> {
        let url = format!("{}/api/tournament", self.base);
        let mut form = vec![
            ("clockTime", (clock_time as f32 / 60f32).to_string()),
            ("clockIncrement", clock_increment.to_string()),
            ("minutes", minutes.to_string()),
        ];
        if let Some(params) = form_params {
            for (key, val) in params.iter() {
                form.push((key, val.to_string()))
            }
        }
        let builder = self.client.post(&url).form(&form);
        self.to_model_full(builder).await
    }

    pub async fn arena_info(&self, id: &str, page: u8) -> LichessResult<ArenaTournament> {
        let url = format!("{}/api/tournament/{}", self.base, id);
        let builder = self.client.get(&url).query(&[("page", page)]);
        self.to_model_full(builder).await
    }

    pub async fn games_by_arena_pgn(
        &self,
        id: &str,
        query_params: Option<Vec<(&str, &str)>>,
    ) -> LichessResult<impl Stream<Item = LichessResult<Bytes>>> {
        let url = format!("{}/api/tournament/{}/games", self.base, id);
        let mut builder = self
            .client
            .get(&url)
            .header("Accept", "application/x-chess-pgn");
        if let Some(params) = query_params {
            builder = builder.query(&params)
        }
        self.to_raw_bytes(builder).await
    }

    pub async fn games_by_arena_json(
        &self,
        id: &str,
        query_params: Option<Vec<(&str, &str)>>,
    ) -> LichessResult<impl Stream<Item = LichessResult<Game>>> {
        let url = format!("{}/api/tournament/{}/games", self.base, id);
        let mut builder = self
            .client
            .get(&url)
            .header("Accept", "application/x-ndjson");
        if let Some(params) = query_params {
            builder = builder.query(&params)
        }
        self.to_model_stream(builder).await
    }

    pub async fn results_by_arena(
        &self,
        id: &str,
        nb_players: u16,
    ) -> LichessResult<impl Stream<Item = LichessResult<PlayerStandings>>> {
        let url = format!("{}/api/tournament/{}/results", self.base, id);
        let builder = self.client.get(&url).query(&[("nb", nb_players)]);
        self.to_model_stream(builder).await
    }

    pub async fn teams_by_arena(&self, id: &str) -> LichessResult<TeamStandings> {
        let url = format!("{}/api/tournament/{}/teams", self.base, id);
        let builder = self.client.get(&url);
        self.to_model_full(builder).await
    }

    pub async fn arenas_by_user(
        &self,
        username: &str,
        nb_tournamensts: u16,
    ) -> LichessResult<impl Stream<Item = LichessResult<ArenaTournament>>> {
        let url = format!("{}/api/user/{}/tournament/created", self.base, username);
        let builder = self.client.get(&url).query(&[("nb", nb_tournamensts)]);
        self.to_model_stream(builder).await
    }
}
