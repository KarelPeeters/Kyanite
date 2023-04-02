//! Structs for account related objects

use chrono::{serde::ts_milliseconds, DateTime, NaiveDate, Utc};
use serde::{de, Deserialize, Deserializer, Serialize};
use serde_with::skip_serializing_none;
use std::collections::HashMap;
use std::convert::TryInto;

/// The different perftypes available for parsing and making rquests
#[allow(missing_docs)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "camelCase")]
pub enum PerfType {
    FromPosition,
    UltraBullet,
    Bullet,
    Blitz,
    Rapid,
    Classical,
    Chess960,
    Crazyhouse,
    Antichess,
    Atomic,
    Horde,
    KingOfTheHill,
    RacingKings,
    ThreeCheck,
    Puzzle,
    Correspondence,
    // Standard,
}

#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UserPerf {
    pub games: Option<u32>,
    pub rating: u16,
    pub rd: Option<u16>,
    #[serde(alias = "progress")]
    pub prog: i32,
    pub prov: Option<bool>,
}

#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct UserProfile {
    pub country: Option<String>,
    pub location: Option<String>,
    pub bio: Option<String>,
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub fide_rating: Option<u16>,
    pub uscf_rating: Option<u16>,
    pub ecf_rating: Option<u16>,
    pub links: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UserPlaytime {
    pub total: u64,
    pub tv: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct UserCount {
    pub all: u32,
    pub rated: u32,
    pub ai: u32,
    pub draw: u32,
    pub draw_h: u32,
    pub loss: u32,
    pub loss_h: u32,
    pub win: u32,
    pub win_h: u32,
    pub bookmark: u32,
    pub playing: u32,
    pub import: u32,
    pub me: u32,
}

/// The all inclusive user object
#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct User {
    pub id: String,
    pub username: String,
    pub closed: Option<bool>,
    pub online: bool,
    pub perfs: HashMap<PerfType, UserPerf>,
    #[serde(deserialize_with = "ts_milliseconds::deserialize")]
    pub created_at: DateTime<Utc>,
    pub disabled: Option<bool>,
    pub engine: Option<bool>,
    pub booster: Option<bool>,
    pub profile: Option<UserProfile>,
    #[serde(deserialize_with = "ts_milliseconds::deserialize")]
    pub seen_at: DateTime<Utc>,
    pub patron: Option<bool>,
    pub play_time: UserPlaytime,
    pub language: Option<String>,
    pub title: Option<String>,
    pub url: Option<String>,
    pub playing: Option<String>,
    pub nb_following: Option<u32>,
    pub nb_followers: Option<u32>,
    pub completion_rate: Option<u8>,
    pub count: Option<UserCount>,
    pub streaming: Option<bool>,
    pub followable: Option<bool>,
    pub following: Option<bool>,
    pub blocking: Option<bool>,
    pub follows_you: Option<bool>,
}

/// Settings of users in non-human-readable(mostly) form
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct UserPreferences {
    pub dark: bool,
    pub transp: bool,
    pub bg_img: String,
    pub is_3d: bool,
    pub theme: String,
    pub piece_set: String,
    pub theme_3d: String,
    pub piece_set_3d: String,
    pub sound_set: String,
    pub blindfold: u8,
    pub auto_queen: u8,
    pub auto_threefold: u8,
    pub takeback: u8,
    pub moretime: u8,
    pub clock_tenths: u8,
    pub clock_bar: bool,
    pub clock_sound: bool,
    pub premove: bool,
    pub animation: u8,
    pub captured: bool,
    pub follow: bool,
    pub highlight: bool,
    pub destination: bool,
    pub coords: u8,
    pub replay: u8,
    pub challenge: u8,
    pub message: u8,
    pub coord_color: u8,
    pub submit_move: u8,
    pub confirm_resign: u8,
    pub insight_share: u8,
    pub keyboard_move: u8,
    pub zen: u8,
    pub move_event: u8,
    pub rook_castle: u8,
}

/// A minimal user object - as received with quite a few endpoint responses
#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LightUser {
    pub id: Option<String>,
    #[serde(alias = "name")]
    pub username: String,
    pub ai: Option<u8>,
    pub perfs: Option<HashMap<PerfType, UserPerf>>,
    pub title: Option<String>,
    pub online: Option<bool>,
    pub playing: Option<bool>,
    pub streaming: Option<bool>,
    pub patron: Option<bool>,
    pub rating: Option<u16>,
    pub provisional: Option<bool>,
    pub lag: Option<u16>,
    #[serde(rename = "gameId")]
    pub game_id: Option<String>, // for simuls
}

/// Was all those years worth it?
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RatingHistory {
    #[serde(rename = "name")]
    pub perf_name: String, // TODO: normalize perf names over the api
    #[serde(deserialize_with = "de_history")]
    #[serde(rename = "points")]
    pub point_history: Vec<(NaiveDate, u16)>,
}

/// Record about user's daily puzzle routine
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct PuzzleActivity {
    pub id: u32,
    #[serde(deserialize_with = "ts_milliseconds::deserialize")]
    pub date: DateTime<Utc>,
    pub rating: u16,
    pub rating_diff: i16,
    pub puzzle_rating: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Matchup {
    pub users: HashMap<String, f32>,
    #[serde(rename = "nbGames")]
    pub nb_games: u32,
}

/// Lifetime record between a pair of users
#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Crosstable {
    pub users: HashMap<String, f32>,
    #[serde(rename = "nbGames")]
    pub nb_games: u32,
    pub matchup: Option<Matchup>,
}

#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OnlineBot {
    pub id: String,
    pub username: String,
    pub title: Option<String>,

    pub play_time: Option<UserPlaytime>,
}

fn de_history<'de, D>(deserializer: D) -> Result<Vec<(NaiveDate, u16)>, D::Error>
where
    D: Deserializer<'de>,
{
    let history: Vec<Vec<u16>> = Vec::deserialize(deserializer)?;
    let mut struggle: Vec<(NaiveDate, u16)> = Vec::new();
    if !history.is_empty() {
        for record in history.iter() {
            struggle.push((
                NaiveDate::from_ymd_opt(
                    record[0].try_into().map_err(de::Error::custom)?,
                    (record[1] + 1).into(), // At lichess months start at 0
                    record[2].into(),
                )
                .unwrap(),
                record[3],
            ))
        }
    }
    Ok(struggle)
}
