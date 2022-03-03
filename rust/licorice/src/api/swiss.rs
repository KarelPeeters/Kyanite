use crate::client::{Lichess, LichessResult};
use crate::models::{
    game::Game,
    tournament::{PlayerStandings, SwissTournament},
};
use bytes::Bytes;
use futures_util::stream::Stream;

impl Lichess {
    pub async fn swiss_new(
        &self,
        team_id: &str,
        clock_time: u16,      // seconds
        clock_increment: u16, // why are theese differnet from arena?
        nb_rounds: u16,
        form_params: Option<&[(&str, &str)]>,
    ) -> LichessResult<SwissTournament> {
        let url = format!("{}/api/swiss/new/{}", self.base, team_id);
        let mut form = vec![
            ("clock.limit", clock_time.to_string()),
            ("clock.increment", clock_increment.to_string()),
            ("nbRounds", nb_rounds.to_string()),
        ];
        if let Some(params) = form_params {
            for (key, val) in params.iter() {
                form.push((key, val.to_string()))
            }
        }
        let builder = self.client.post(&url).form(&form);
        self.to_model_full(builder).await
    }

    pub async fn swiss_trf(&self, id: &str) -> LichessResult<String> {
        let url = format!("{}/swiss/{}.trf", self.base, id);
        let builder = self.client.get(&url);
        self.to_raw_str(builder).await
    }

    pub async fn games_by_swiss_pgn(
        &self,
        id: &str,
        query_params: Option<Vec<(&str, &str)>>,
    ) -> LichessResult<impl Stream<Item = LichessResult<Bytes>>> {
        let url = format!("{}/api/swiss/{}/games", self.base, id);
        let mut builder = self.client.get(&url).header("Accept", "application/x-chess-pgn");
        if let Some(params) = query_params {
            builder = builder.query(&params)
        }
        self.to_raw_bytes(builder).await
    }

    pub async fn games_by_swiss_json(
        &self,
        id: &str,
        query_params: Option<Vec<(&str, &str)>>,
    ) -> LichessResult<impl Stream<Item = LichessResult<Game>>> {
        let url = format!("{}/api/swiss/{}/games", self.base, id);
        let mut builder = self.client.get(&url).header("Accept", "application/x-ndjson");
        if let Some(params) = query_params {
            builder = builder.query(&params)
        }
        self.to_model_stream(builder).await
    }

    pub async fn results_by_swiss(
        &self,
        id: &str,
        nb_players: u16,
    ) -> LichessResult<impl Stream<Item = LichessResult<PlayerStandings>>> {
        let url = format!("{}/api/swiss/{}/results", self.base, id);
        let builder = self.client.get(&url).query(&[("nb", nb_players)]);
        self.to_model_stream(builder).await
    }
}
