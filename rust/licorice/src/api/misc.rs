use crate::client::{Lichess, LichessResult};
use crate::models::{game::Eval, tournament::Simuls};
use serde_json::{from_value, Value};

impl Lichess {
    pub async fn get_current_simuls(&self) -> LichessResult<Simuls> {
        let url = format!("{}/api/simul", self.base);
        let builder = self.client.get(&url);
        self.to_model_full(builder).await
    }

    pub async fn study_chapter_pgn(
        &self,
        study_id: &str,
        chapter_id: &str,
        query_params: Option<&[(&str, &str)]>,
    ) -> LichessResult<String> {
        let url = format!("{}/study/{}/{}.pgn", self.base, study_id, chapter_id);
        let mut builder = self.client.get(&url);
        if let Some(params) = query_params {
            builder = builder.query(&params)
        }
        self.to_raw_str(builder).await
    }

    pub async fn study_full_pgn(&self, study_id: &str, query_params: Option<&[(&str, &str)]>) -> LichessResult<String> {
        let url = format!("{}/study/{}.pgn", self.base, study_id);
        let mut builder = self.client.get(&url);
        if let Some(params) = query_params {
            builder = builder.query(&params)
        }
        self.to_raw_str(builder).await
    }

    pub async fn mesage(&self, recipient: &str, message: &str) -> LichessResult<()> {
        let url = format!("{}/inbox/{}", self.base, recipient);
        let builder = self.client.post(&url).form(&[("text", message)]);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }

    pub async fn get_cloud_eval(&self, fen: &str, multi_pv: u8, variant: Option<&str>) -> LichessResult<Eval> {
        let url = format!("{}/api/cloud-eval", self.base);
        let builder = self.client.get(&url).query(&[
            ("fen", fen),
            ("multiPv", &multi_pv.to_string()),
            ("variant", variant.unwrap_or("standard")),
        ]);
        self.to_model_full(builder).await
    }
}
