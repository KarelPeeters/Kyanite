//! Was this module really necessary?
use super::user::LightUser;
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;

/// Information about a lichess team
/// Conisder joining out discord
/// and then our team :-)
#[skip_serializing_none]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Team {
    pub id: String,
    pub name: String,
    pub description: String,
    pub open: bool,
    pub leader: LightUser,
    pub leaders: Vec<LightUser>,
    #[serde(rename = "nbMembers")]
    pub nb_members: u32,
    pub location: Option<String>,
    pub joined: Option<bool>,
    pub requested: Option<bool>,
}
