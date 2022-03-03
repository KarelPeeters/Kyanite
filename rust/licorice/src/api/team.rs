use crate::client::{Lichess, LichessResult};
use crate::models::team::Team;
use crate::models::tournament::{ArenaTournament, SwissTournament};
use futures_util::stream::Stream;
use serde_json::{from_value, Value};

impl Lichess {
    pub async fn get_team_swiss_tournaments(
        &self,
        team_id: &str,
        max: u16,
    ) -> LichessResult<impl Stream<Item = LichessResult<SwissTournament>>> {
        let url = format!("{}/api/team/{}/swiss", self.base, team_id);
        let builder = self.client.get(&url).query(&[("max", max)]);
        self.to_model_stream(builder).await
    }

    pub async fn get_a_single_team(&self, team_id: &str) -> LichessResult<Team> {
        let url = format!("{}/api/team/{}", self.base, team_id);
        let builder = self.client.get(&url);
        self.to_model_full(builder).await
    }
    pub async fn get_popular_teams(&self, page: Option<u16>) -> LichessResult<Vec<Team>> {
        let url = format!("{}/api/team/all", self.base);
        let builder = self.client.get(&url).query(&[("page", page.unwrap_or(1))]);
        let current_page_results_json = self.to_model_full::<Value>(builder);
        from_value(current_page_results_json.await?["currentPageResults"].take())
            .map_err(Into::into)
    }

    pub async fn teams_of_a_player(&self, username: &str) -> LichessResult<Vec<Team>> {
        let url = format!("{}/api/team/of/{}", self.base, username);
        let builder = self.client.get(&url);
        self.to_model_full(builder).await
    }

    pub async fn search_teams(&self, text: &str, page: Option<u16>) -> LichessResult<Vec<Team>> {
        let url = format!("{}/api/team/search", self.base);
        let builder = self
            .client
            .get(&url)
            .query(&[("text", text), ("page", &page.unwrap_or(1).to_string())]);
        let current_page_results_json = self.to_model_full::<Value>(builder);
        from_value(current_page_results_json.await?["currentPageResults"].take())
            .map_err(Into::into)
    }

    pub async fn get_team_arena_tournaments(
        &self,
        team_id: &str,
        max: u16,
    ) -> LichessResult<impl Stream<Item = LichessResult<ArenaTournament>>> {
        let url = format!("{}/api/team/{}/arena", self.base, team_id);
        let builder = self.client.get(&url).query(&[("max", max)]);
        self.to_model_stream(builder).await
    }

    pub async fn join_a_team(&self, team_id: &str, message: Option<&str>) -> LichessResult<()> {
        let url = format!("{}/team/{}/join", self.base, team_id);
        let mut builder = self.client.post(&url);
        if let Some(msg) = message {
            builder = builder.form(&[("message", msg)])
        }
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    pub async fn leave_a_team(&self, team_id: &str) -> LichessResult<()> {
        let url = format!("{}/team/{}/quit", self.base, team_id);
        let builder = self.client.post(&url);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    pub async fn kick_user_from_team(&self, team_id: &str, user_id: &str) -> LichessResult<()> {
        let url = format!("{}/team/{}/kick/{}", self.base, team_id, user_id);
        let builder = self.client.post(&url);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    pub async fn message_all_members(&self, team_id: &str, message: &str) -> LichessResult<()> {
        let url = format!("{}/team/{}/pm-all", self.base, team_id);
        let builder = self.client.post(&url).form(&[("message", message)]);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }
}
