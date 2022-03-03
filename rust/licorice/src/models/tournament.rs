//! Tournament and simul objects

use super::game::{Clock, Variant};
use super::user::LightUser;
use chrono::{DateTime, Duration, TimeZone, Utc};
use serde::{de, Deserialize, Deserializer, Serialize};
use serde_with::{serde_as, skip_serializing_none, DurationSeconds};
use std::collections::HashMap;
use std::fmt;

#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Position {
    pub eco: Option<String>,
    pub name: String,
    #[serde(rename = "wikiPath")]
    pub wiki_path: Option<String>,
    pub fen: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Schedule {
    pub freq: String,
    pub speed: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeamBattle1 {
    pub teams: HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeamBattle2 {
    pub teams: Vec<String>,
    #[serde(rename = "nbLeaders")]
    pub nb_leaders: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(untagged)]
pub enum TeamBattle {
    TeamBattle1(TeamBattle1),
    TeamBattle2(TeamBattle2),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(untagged)]
pub enum ArenaVariant {
    MapVariant(Variant),
    StrVariant(String),
}

/// Information about arena tournaments (includes team battles)
#[serde_as]
#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct ArenaTournament {
    pub id: String,
    pub created_by: String,
    pub system: String,
    pub minutes: u16,
    pub clock: Clock,
    pub rated: Option<bool>,
    pub description: Option<String>,
    pub full_name: String,
    pub nb_players: u16,
    pub quote: Option<Quote>,
    pub variant: ArenaVariant,
    pub great_player: Option<GreatPlayer>,
    pub berserkable: Option<bool>,
    pub seconds_to_start: Option<u32>,
    pub seconds_to_finish: Option<u32>,
    pub is_started: Option<bool>,
    pub is_finished: Option<bool>,
    pub is_recently_finished: Option<bool>,
    pub pairings_closed: Option<bool>,
    pub has_max_rating: Option<bool>,
    #[serde(default, deserialize_with = "de_time")]
    pub starts_at: Option<DateTime<Utc>>,
    #[serde(default, deserialize_with = "de_time")]
    pub finishes_at: Option<DateTime<Utc>>,
    pub duels: Option<Vec<Duel>>,
    pub standing: Option<Standings>, // yep there's a doc mismatch!
    pub featured: Option<Featured>,
    pub podium: Option<Vec<PlayerStandings>>,
    pub stats: Option<TournamentStats>,
    pub verdicts: Option<Verdicts>,
    pub status: Option<u8>,
    pub spotlight: Option<Spotlight>,
    pub position: Option<Position>,
    pub private: Option<bool>,
    pub schedule: Option<Schedule>,
    pub team_battle: Option<TeamBattle>,
    pub duel_teams: Option<HashMap<String, String>>,
    pub team_standing: Option<Vec<TeamResult>>,
    pub winner: Option<LightUser>,
}

fn de_time<'de, D>(deserializer: D) -> Result<Option<DateTime<Utc>>, D::Error>
where
    D: Deserializer<'de>,
{
    struct TimeVisitor;

    impl<'de> de::Visitor<'de> for TimeVisitor {
        type Value = Option<DateTime<Utc>>;

        fn expecting(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt.write_str("integer or string")
        }

        fn visit_u64<E>(self, val: u64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(Some(Utc.timestamp_millis(val as i64)))
        }
        fn visit_str<E>(self, val: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            // let dt = DateTime::parse_from_rfc3339(val)?;
            Ok(Some(DateTime::parse_from_rfc3339(val).unwrap().with_timezone(&Utc)))
        }
    }
    deserializer.deserialize_any(TimeVisitor)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spotlight {
    pub headline: String,
    pub description: String,
    #[serde(rename = "iconFont")]
    pub icon_font: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Verdicts {
    pub list: Vec<Verdict>,
    pub accepted: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Verdict {
    pub condition: String,
    pub verdict: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Duel {
    pub id: String,
    #[serde(rename = "p")]
    pub duelists: Vec<Duelist>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Duelist {
    #[serde(rename = "n")]
    pub name: String,
    #[serde(rename = "r")]
    pub rating: u16,
    #[serde(rename = "k")]
    pub rank: u16,
    #[serde(rename = "t")]
    pub title: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Quote {
    pub text: String,
    pub author: String,
}

#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NextRound {
    pub at: DateTime<Utc>,
    #[serde_as(as = "DurationSeconds<i64>")]
    pub r#in: Duration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GreatPlayer {
    pub name: String,
    pub url: String,
}

/// Information about swiss tournaments
#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct SwissTournament {
    pub id: String,
    pub created_by: String,
    pub starts_at: DateTime<Utc>,
    pub name: String,
    pub clock: Clock,
    pub variant: String,
    pub round: u8,
    pub nb_rounds: u8,
    pub nb_players: u16,
    pub nb_ongoing: u16,
    pub status: String,
    pub quote: Option<Quote>,
    pub next_round: Option<NextRound>,
    pub great_player: Option<GreatPlayer>,
    pub rated: bool,
}

/// Timetable record for arena tournaments
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArenaSchedule {
    pub created: Vec<ArenaTournament>,
    pub started: Vec<ArenaTournament>,
    pub finished: Vec<ArenaTournament>,
}

#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sheet {
    pub scores: Vec<Score>,
    pub total: u16,
    pub fire: Option<bool>,
}

#[skip_serializing_none]
#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Score {
    pub value: u8,
    pub part_of_streak: Option<u8>,
}

struct ScoreDeserializer;

impl<'de> de::Visitor<'de> for ScoreDeserializer {
    type Value = Score;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("number or list of numbers")
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let value = value as u8;
        let part_of_streak = None;
        Ok(Score { value, part_of_streak })
    }

    fn visit_seq<S>(self, mut visitor: S) -> Result<Self::Value, S::Error>
    where
        S: de::SeqAccess<'de>,
    {
        let value = visitor.next_element::<u8>()?.unwrap();
        let part_of_streak = visitor.next_element::<u8>()?;

        Ok(Score { value, part_of_streak })
    }
}

impl<'de> Deserialize<'de> for Score {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(ScoreDeserializer)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct TournamentStats {
    pub games: u32,
    pub moves: u32,
    pub white_wins: u32,
    pub black_wins: u32,
    pub draws: u32,
    pub berserks: u32,
    pub average_rating: u16,
}

#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Featured {
    pub id: String,
    pub fen: String,
    pub orientation: Option<String>,
    pub color: String,
    #[serde(rename = "lastMove")]
    pub last_move: String,
    pub white: FeaturedPlayer,
    pub black: FeaturedPlayer,
    #[serde(rename = "c")]
    pub remaining_secs: RemainingSecs,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemainingSecs {
    pub white: u16,
    pub black: u16,
}

#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaturedPlayer {
    pub rank: u8,
    pub name: String,
    pub rating: u16,
    pub title: Option<String>,
    pub berserk: Option<bool>,
}

/// Report card for tournament participants
#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayerStandings {
    pub rank: u16,
    pub score: u32,
    #[serde(rename = "tieBreak")]
    pub tiebreak: Option<f32>,
    pub rating: u16,
    #[serde(alias = "name")]
    pub username: String,
    pub sheet: Option<Sheet>,
    pub title: Option<String>,
    pub provisional: Option<bool>,
    pub withdraw: Option<bool>,
    pub absent: Option<bool>,
    #[serde(rename = "nb")]
    pub counts: Option<Counts>,
    pub performance: Option<u16>,
    pub team: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counts {
    pub game: u16,
    pub berserk: u16,
    pub win: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Standings {
    pub page: u8,
    pub players: Vec<PlayerStandings>,
}

#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayerResult {
    pub user: LightUser,
    pub score: Option<u16>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeamResult {
    pub rank: u8,
    pub id: String,
    pub score: u16,
    pub players: Vec<PlayerResult>,
}

/// How did the battle go?
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TeamStandings {
    pub id: String,
    pub teams: Vec<TeamResult>,
}

/// Records of current simultaneous exhibitions
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "camelCase")]
pub struct Simul {
    pub full_name: String,
    pub host: LightUser,
    pub id: String,
    pub is_created: bool,
    pub is_finished: bool,
    pub is_running: bool,
    pub name: String,
    pub nb_applicants: u16,
    pub nb_pairings: u8,
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Simuls {
    pub pending: Vec<Simul>,
    pub created: Vec<Simul>,
    pub started: Vec<Simul>,
    pub finished: Vec<Simul>,
}
