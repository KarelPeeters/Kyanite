use crate::client::{Lichess, LichessResult};
use crate::models::board::{ChallengeGame, EntityChallenge, OpenChallenges};
use serde_json::{from_value, Value};

impl Lichess {
    pub async fn challenge_create(
        &self,
        username: &str,
        form_params: Option<&[(&str, &str)]>,
    ) -> LichessResult<EntityChallenge> {
        let url = format!("{}/api/challenge/{}", self.base, username);
        let mut builder = self.client.post(&url);
        if let Some(params) = form_params {
            builder = builder.form(&params)
        }
        self.to_model_full(builder).await
    }

    pub async fn challenge_accept(&self, challenge_id: &str) -> LichessResult<()> {
        let url = format!("{}/api/challenge/{}/accept", self.base, challenge_id);
        let builder = self.client.post(&url);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    pub async fn challenge_decline(&self, challenge_id: &str, reason: Option<&str>) -> LichessResult<()> {
        let url = format!("{}/api/challenge/{}/decline", self.base, challenge_id);
        let form = vec![("reason", reason.map_or("generic".to_string(), String::from))];
        let builder = self.client.post(&url).form(&form);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    pub async fn challenge_cancel(&self, challenge_id: &str) -> LichessResult<()> {
        let url = format!("{}/api/challenge/{}/cancel", self.base, challenge_id);
        let builder = self.client.post(&url);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    pub async fn challenge_stockfish(
        &self,
        level: u8,
        form_params: Option<&[(&str, &str)]>,
    ) -> LichessResult<ChallengeGame> {
        let url = format!("{}/api/challenge/ai", self.base);
        let mut form = vec![("level", level.to_string())];
        if let Some(params) = form_params {
            for (key, val) in params.iter() {
                form.push((key, val.to_string()))
            }
        }
        let builder = self.client.post(&url).form(&form);
        self.to_model_full(builder).await
    }

    pub async fn challenge_open(
        &self,
        form_params: Option<&[(&str, &str)]>,
    ) -> LichessResult<EntityChallenge> {
        let url = format!("{}/api/challenge/open", self.base);
        let mut builder = self.client.post(&url);
        if let Some(params) = form_params {
            builder = builder.form(&params)
        }
        self.to_model_full(builder).await
    }

    pub async fn start_game_clocks(
        &self,
        game_id: &str,
        token1: &str,
        token2: &str,
    ) -> LichessResult<()> {
        let url = format!("{}/api/challenge/{}/start-clocks", self.base, game_id);
        let builder = self
            .client
            .post(&url)
            .query(&[("token1", token1), ("token2", token2)]);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    pub async fn admin_challenge(
        &self,
        user1: &str,
        user2: &str,
        form_params: Option<&[(&str, &str)]>,
    ) -> LichessResult<EntityChallenge> {
        let url = format!("{}/api/challenge/admin/{}/{}", self.base, user1, user2);
        let mut builder = self.client.post(&url);
        if let Some(params) = form_params {
            builder = builder.form(&params)
        }
        self.to_model_full(builder).await
    }

    pub async fn list_challenges(&self) -> LichessResult<OpenChallenges> {
        let url = format!("{}/api/challenge", self.base);
        let builder = self.client.get(url);
        self.to_model_full(builder).await
    }
}
