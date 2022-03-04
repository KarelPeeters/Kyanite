use crate::client::{Lichess, LichessResult};
use crate::models::user::User;
use futures_util::stream::Stream;

impl Lichess {
    pub async fn get_followings(&self, username: &str) -> LichessResult<impl Stream<Item = LichessResult<User>>> {
        let url = format!("{}/api/user/{}/following", self.base, username);
        let builder = self.client.get(&url);
        self.to_model_stream(builder).await
    }

    pub async fn get_followers(&self, username: &str) -> LichessResult<impl Stream<Item = LichessResult<User>>> {
        let url = format!("{}/api/user/{}/followers", self.base, username);
        let builder = self.client.get(&url);
        self.to_model_stream(builder).await
    }
}
