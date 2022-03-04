//! Structures to investogate errors from API

use reqwest::{Response, StatusCode};
use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
/// Encompasses all possible errors emitted by the client
pub enum LichessError {
    /// Error when the limit is reached
    #[error("exceeded request limit")]
    RateLimited(Option<usize>),
    /// Propagated errors from the reqwest crate
    #[error("request error: {0}")]
    Request(#[from] reqwest::Error),
    /// Errors for when server returns non 200 response
    #[error("status code {0}: {1}")]
    StatusCode(u16, String),
    /// If the API has a designated error message for the request
    #[error("lichess error: {0}")]
    API(#[from] APIError),
    /// (de)serializing and related errors
    #[error("json parse error: {0}")]
    ParseJSON(#[from] serde_json::Error),
    /// Errors while reading buffers
    #[error("input/output error: {0}")]
    IO(#[from] std::io::Error),
}

impl LichessError {
    pub(crate) async fn from_response(response: Response) -> Self {
        match response.status() {
            StatusCode::TOO_MANY_REQUESTS => Self::RateLimited(
                response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|header| header.to_str().ok())
                    .and_then(|duration| duration.parse().ok()),
            ),
            status => response
                .json::<APIError>()
                .await
                .map(Into::into)
                .unwrap_or_else(|_| status.into()),
        }
    }
}

impl From<StatusCode> for LichessError {
    fn from(code: StatusCode) -> Self {
        Self::StatusCode(code.as_u16(), code.canonical_reason().unwrap_or("unknown").to_string())
    }
}

/// Error response objects returned by the server
#[derive(Debug, Error, Deserialize)]
#[error("error: {error}")]
pub struct APIError {
    error: String,
}
