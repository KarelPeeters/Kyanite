use crate::client::{Lichess, LichessResult};
use crate::models::broadcast::Broadcast;
use futures_util::stream::Stream;
use serde_json::{from_value, Value};

impl Lichess {
    pub async fn get_official_broadcasts(
        &self,
        nb_broadcasts: u8,
    ) -> LichessResult<impl Stream<Item = LichessResult<Broadcast>>> {
        let url = format!("{}/api/broadcast", self.base,);
        let builder = self.client.get(&url).query(&[("nb", nb_broadcasts)]);
        self.to_model_stream(builder).await
    }

    pub async fn create_broadcast(
        &self,
        name: &str,
        description: &str,
        form_params: Option<&[(&str, &str)]>,
    ) -> LichessResult<Broadcast> {
        let url = format!("{}/broadcast/new", self.base);
        let mut form = vec![("name", name), ("description", description)];
        if let Some(params) = form_params {
            for (key, val) in params.iter() {
                form.push((key, *val))
            }
        }
        let builder = self.client.post(&url).form(&form);
        self.to_model_full(builder).await
    }

    pub async fn get_broadcast(&self, broadcast_id: &str) -> LichessResult<Broadcast> {
        let url = format!("{}/broadcast/-/{}", self.base, broadcast_id);
        let builder = self.client.get(&url);
        self.to_model_full(builder).await
    }

    pub async fn update_broadcast(
        &self,
        broadcast_id: &str,
        name: &str,
        description: &str,
        form_params: Option<&[(&str, &str)]>,
    ) -> LichessResult<Broadcast> {
        let url = format!("{}/broadcast/-/{}/edit", self.base, broadcast_id);
        let mut form = vec![("name", name), ("description", description)];
        if let Some(params) = form_params {
            for (key, val) in params.iter() {
                form.push((key, *val))
            }
        }
        let builder = self.client.post(&url).form(&form);
        self.to_model_full(builder).await
    }

    pub async fn push_to_broadcast(&self, broadcast_id: &str, pgn: &str) -> LichessResult<()> {
        let url = format!("{}/broadcast/-/{}/push", self.base, broadcast_id);
        let builder = self.client.post(&url).body(pgn.to_owned());
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }
}
