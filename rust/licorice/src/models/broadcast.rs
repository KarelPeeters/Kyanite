use chrono::{serde::ts_milliseconds_option, DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;

#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct Broadcast {
    pub id: String,
    pub url: String,
    pub name: String,
    pub description: String,
    pub credit: Option<String>,
    pub markup: Option<String>,
    #[serde(default, deserialize_with = "ts_milliseconds_option::deserialize")]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(default, deserialize_with = "ts_milliseconds_option::deserialize")]
    pub starts_at: Option<DateTime<Utc>>,
    pub official: Option<bool>,
    pub finished: Option<bool>,
    pub markdown: Option<String>,
    pub throttle: Option<u16>,
    pub sync: Option<Sync>,
}

#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sync {
    pub ongoing: bool,
    pub log: Vec<String>,
    pub url: Option<String>,
}
