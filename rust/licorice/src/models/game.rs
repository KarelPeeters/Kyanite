//! Game and analysis realted objects

use super::user::{LightUser, PerfType};
use chrono::{
    serde::{ts_milliseconds, ts_milliseconds_option},
    DateTime, Utc,
};
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;

#[allow(missing_docs)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayerAnalysis {
    pub inaccuracy: u16,
    pub mistake: u16,
    pub blunder: u16,
    pub acpl: u16,
}

#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct Entity {
    pub user: Option<LightUser>,
    pub user_id: Option<String>,
    pub rating: u16,
    pub rating_diff: Option<i16>,
    pub provisional: Option<bool>,
    pub analysis: Option<PlayerAnalysis>,
}

#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StockFish {
    #[serde(rename = "aiLevel")]
    pub ai_level: u8,
    pub analysis: Option<PlayerAnalysis>,
}

/// Two types of opponents one can face
#[allow(missing_docs)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Player {
    Entity(Entity),
    StockFish(StockFish),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Players {
    pub white: Player,
    pub black: Player,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Opening {
    pub eco: String,
    pub name: String,
    pub ply: u16,
}

/// All possible fields of information attached to 'clock' in response
#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct Clock {
    pub initial: Option<u32>,
    pub increment: Option<u16>,
    pub total_time: Option<u16>,
    pub limit: Option<u16>,
    pub days_per_turn: Option<u8>,
    pub show: Option<String>,
    pub r#type: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Judgement {
    pub name: String,
    pub comment: String,
}

#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoveAnalysis {
    pub mate: Option<u8>,
    pub eval: Option<i16>, // in the JSON export the eval values are 100*(engine eval)
    pub best: Option<String>,
    pub variation: Option<String>,
    pub judgment: Option<Judgement>,
}

/// The OG game struct
#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct Game {
    pub id: String,
    pub rated: bool,
    pub variant: String,
    pub speed: String,
    pub perf: PerfType,
    #[serde(deserialize_with = "ts_milliseconds::deserialize")]
    pub created_at: DateTime<Utc>,
    #[serde(default, deserialize_with = "ts_milliseconds_option::deserialize")]
    pub last_move_at: Option<DateTime<Utc>>,
    pub status: String,
    pub players: Players,
    pub initial_fen: Option<String>,
    pub winner: Option<String>, // white or black
    pub opening: Option<Opening>,
    pub moves: Option<String>,
    pub pgn: Option<String>,
    pub days_per_turn: Option<u8>,
    pub analysis: Option<Vec<MoveAnalysis>>,
    pub tournament: Option<String>,
    pub clock: Option<Clock>,
}

/// Provdies information to identify standard and non-standard chess
#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub key: String,
    pub short: Option<String>,
    pub name: String,
}

/// Record of different time controls in which games can occur
#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Perf {
    pub icon: Option<String>,
    pub key: Option<String>,
    pub name: String,
    pub position: Option<u8>,
}

/// Game information from one of the player's perspective
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct UserGame {
    pub full_id: String,
    pub game_id: String,
    pub fen: String,
    pub color: String,
    pub has_moved: bool,
    pub last_move: String,
    pub variant: Variant,
    pub speed: String,
    pub perf: PerfType,
    pub rated: bool,
    pub opponent: LightUser,
    pub is_my_turn: bool,
    pub seconds_left: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Channel {
    pub user: LightUser,
    pub rating: u16,
    #[serde(rename = "gameId")]
    pub game_id: String,
}

/// Lichess TV live status
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "PascalCase")]
pub struct TVChannels {
    pub bot: Channel,
    pub blitz: Channel,
    #[serde(rename = "Racing Kings")]
    pub racing_kings: Channel,
    pub ultra_bullet: Channel,
    pub bullet: Channel,
    pub classical: Channel,
    #[serde(rename = "Three-check")]
    pub three_check: Channel,
    pub antichess: Channel,
    pub computer: Channel,
    pub horde: Channel,
    pub rapid: Channel,
    pub atomic: Channel,
    pub crazyhouse: Channel,
    pub chess960: Channel,
    #[serde(rename = "King of the Hill")]
    pub king_of_the_hill: Channel,
    #[serde(rename = "Top Rated")]
    pub top_rated: Channel,
}

#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PV {
    pub moves: String,
    pub cp: i16,
}

/// Server analysis of a position
#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Eval {
    pub fen: String,
    pub knodes: u16,
    pub depth: u8,
    pub pvs: Vec<PV>,
}
