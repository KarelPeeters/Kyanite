use crate::client::{Lichess, LichessResult};
use crate::models::user::{Crosstable, LightUser, PerfType, PuzzleActivity, RatingHistory, User};
use futures_util::stream::Stream;
use serde_json::{from_value, to_string, Value};
use std::collections::HashMap;

/// Endpoints which yield information regarding geneal lichess users
impl Lichess {
    /// Provides basic information about a set of lichess users.
    /// At most 50 ids are entertained in ech request.
    ///
    /// Parameters:
    /// - user_ids - a reference to a `Vec` or `array` of user ids, i.e., usernames
    /// in lowercase
    /// [Reference](https://lichess.org/api#operation/apiUsersStatus)
    pub async fn get_user_status(&self, user_ids: &[&str]) -> LichessResult<Vec<LightUser>> {
        let url = format!("{}{}", self.base, "/api/users/status");
        let builder = self.client.get(&url).query(&[("ids", user_ids.join(","))]);
        self.to_model_full(builder).await
    }

    /// Returns bsic information about top 10 players for each speed and variant.
    /// In the returned [HashMap](std::collections::HashMap),for a particular
    /// [PerfType](crate::models::user::PerfType) each
    /// [LightUser](crate::models::user::LightUser) will contain that
    /// [PerfType](crate::models::user::PerfType) only in their `perfs` field
    ///
    /// [Reference](https://lichess.org/api#operation/player)
    pub async fn get_all_top_10(&self) -> LichessResult<HashMap<PerfType, Vec<LightUser>>> {
        let url = format!("{}{}", self.base, "/player");
        let builder = self
            .client
            .get(&url)
            .header("Accept", "application/vnd.lichess.v3+json");
        self.to_model_full(builder).await
    }

    /// Gets a [Vec](std::Vec) of users who tops the leaderboard for the
    /// provided [PerfType](`crate::models::user::PerfType`)
    /// Parameters:
    /// - nb_users - Number of top players to fetch from the leaderbaord.
    /// Must be at most 200, the capacty of each lederboard.
    /// [Reference](https://lichess.org/api#operation/playerTopNbPerfType)
    pub async fn get_one_leaderboard(&self, nb_users: u8, perf_type: PerfType) -> LichessResult<Vec<LightUser>> {
        let mut perf_str = to_string(&perf_type)?;
        // hack as representations of enum variants are enclosed with ""
        perf_str = perf_str[1..perf_str.len() - 1].to_owned();
        let url = format!("{}{}/{}/{}", self.base, "/player/top", nb_users, perf_str);
        let builder = self
            .client
            .get(&url)
            .header("Accept", "application/vnd.lichess.v3+json");
        let users_json = self.to_model_full::<Value>(builder);
        from_value(users_json.await?["users"].take()).map_err(Into::into)
    }

    /// Entire(?) public information regarding a lichess user
    /// Parameters:
    /// -username - case-insensitive ID of a lichess user
    /// [Reference](https://lichess.org/api#operation/apiUser)
    pub async fn get_user_public(&self, username: &str) -> LichessResult<User> {
        let url = format!("{}{}/{}", self.base, "/api/user", username);
        let builder = self.client.get(&url);
        self.to_model_full(builder).await
    }

    /// Records of a user's rating across mutiple perfs
    /// Parameters:
    /// -username - case-insensitive ID of a lichess user
    /// [Reference](https://lichess.org/api#operation/apiUserRatingHistory)
    pub async fn get_rating_history(&self, username: &str) -> LichessResult<Vec<RatingHistory>> {
        let url = format!("{}{}/{}/rating-history", self.base, "/api/user", username);
        let builder = self.client.get(&url);
        self.to_model_full(builder).await
    }

    // TODO: fn get_user-activity()

    /// Puzzle activity of the authenticated user
    pub async fn get_my_puzzle_activity(
        &self,
        max: Option<u64>,
    ) -> LichessResult<impl Stream<Item = LichessResult<PuzzleActivity>>> {
        let query: String;
        match max {
            Some(val) => query = format!("{}", val),
            None => query = String::new(),
        }
        let url = format!("{}{}?max={}", self.base, "/api/user/puzzle-activity", query);
        let builder = self.client.get(&url);
        self.to_model_stream(builder).await
    }

    /// Informations of upto 300 users
    pub async fn get_users_by_ids(&self, ids: &[&str]) -> LichessResult<Vec<User>> {
        let url = format!("{}{}", self.base, "/api/users");
        let builder = self.client.post(&url).body(ids.join(","));
        self.to_model_full(builder).await
    }

    /// Returns a stream of members belonging to the provided team
    pub async fn get_members_of_a_team(&self, team_id: &str) -> LichessResult<impl Stream<Item = LichessResult<User>>> {
        let url = format!("{}{}/{}/users", self.base, "/api/team", team_id);
        let builder = self.client.get(&url);
        self.to_model_stream(builder).await
    }

    /// Basic information about users currently streaming in lichess
    pub async fn get_live_streamers(&self) -> LichessResult<Vec<LightUser>> {
        let url = format!("{}/streamer/live", self.base);
        let builder = self.client.get(&url);
        self.to_model_full(builder).await
    }

    /// Returns records of how two players match up against each other
    pub async fn get_crosstable(&self, player: &str, opponent: &str, matchup: bool) -> LichessResult<Crosstable> {
        let url = format!("{}/api/crosstable/{}/{}", self.base, player, opponent);
        let builder = self.client.get(&url).query(&[("matchup", matchup)]);
        self.to_model_full(builder).await
    }
}
