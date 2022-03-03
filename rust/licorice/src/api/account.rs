use crate::client::{Lichess, LichessResult};
use crate::models::user::{User, UserPreferences};
use serde_json::{from_value, Value};

impl Lichess {
    pub async fn get_my_profile(&self) -> LichessResult<User> {
        let url = format!("{}{}", self.base, "/api/account");
        let builder = self.client.get(&url);
        self.to_model_full(builder).await
    }

    pub async fn get_my_email_address(&self) -> LichessResult<String> {
        let url = format!("{}{}", self.base, "/api/account/email");
        let builder = self.client.get(&url);
        let email_json = self.to_model_full::<Value>(builder);
        from_value(email_json.await?["email"].take()).map_err(Into::into)
    }

    pub async fn get_my_preferences(&self) -> LichessResult<UserPreferences> {
        let url = format!("{}{}", self.base, "/api/account/preferences");
        let builder = self.client.get(&url);
        let prefs_json = self.to_model_full::<Value>(builder);
        from_value(prefs_json.await?["prefs"].take()).map_err(Into::into)
    }

    pub async fn get_my_kid_mode_status(&self) -> LichessResult<bool> {
        let url = format!("{}{}", self.base, "/api/account/kid");
        let builder = self.client.get(&url);
        let kid_json = self.to_model_full::<Value>(builder);
        from_value(kid_json.await?["kid"].take()).map_err(Into::into)
    }

    pub async fn set_my_kid_mode_status(&self, value: bool) -> LichessResult<()> {
        let url = format!("{}{}", self.base, "/api/account/kid");
        let builder = self.client.post(&url).query(&[("v", value)]);
        let ok_json = self.to_model_full::<Value>(builder);
        assert!(from_value::<bool>(ok_json.await?["ok"].take())?);
        Ok(())
    }
}
