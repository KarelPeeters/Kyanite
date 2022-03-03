//! Provides the base client object and Result type

use super::errors::LichessError;
use futures_util::stream::{Stream, StreamExt, TryStreamExt};
use reqwest::{header, Client, RequestBuilder, Response};
use serde::de::DeserializeOwned;
use serde_json::from_str;
use std::io::{Error, ErrorKind};
use tokio::io::AsyncBufReadExt;
use tokio_stream::wrappers::LinesStream;
use tokio_util::io::StreamReader;

/// `Result` type for [`LichessError`]
pub type LichessResult<T> = Result<T, LichessError>;

/// The base client onject
pub struct Lichess {
    pub(crate) client: Client,
    pub(crate) base: String,
}

impl Lichess {
    /// Create an unathenticaed instance of the client
    /// Even with endpoints what do not require OAuth, this will be imposed with constrints
    pub fn default() -> Lichess {
        Lichess {
            client: Client::new(),
            base: String::from("https://lichess.org"),
        }
    }

    /// Creates an authenticaed instance using the provided token
    pub fn new(pat: String) -> Lichess {
        let mut headers = header::HeaderMap::new();
        let mut header_value = header::HeaderValue::from_str(&format!("Bearer {}", pat)).unwrap();
        header_value.set_sensitive(true);
        headers.insert(header::AUTHORIZATION, header_value);
        Lichess {
            client: reqwest::Client::builder().default_headers(headers).build().unwrap(),
            base: String::from("https://lichess.org"),
        }
    }

    async fn api_call(&self, builder: RequestBuilder) -> LichessResult<Response> {
        let response = builder.send().await.map_err(LichessError::from)?;
        if response.status().is_success() {
            Ok(response)
        } else {
            Err(LichessError::from_response(response).await)
        }
    }

    pub(crate) async fn to_raw_str(&self, builder: RequestBuilder) -> LichessResult<String> {
        self.api_call(builder).await?.text().await.map_err(Into::into)
    }

    pub(crate) async fn to_raw_bytes(
        &self,
        builder: RequestBuilder,
    ) -> LichessResult<impl Stream<Item = LichessResult<bytes::Bytes>>> {
        Ok(self.api_call(builder).await?.bytes_stream().map_err(Into::into))
    }

    pub(crate) async fn to_model_full<T: DeserializeOwned>(&self, builder: RequestBuilder) -> LichessResult<T> {
        // self.api_call(builder).await?.json::<T>().await.map_err(Into::into)
        // https://github.com/serde-rs/json/issues/160
        from_str(&self.api_call(builder).await?.text().await?).map_err(Into::into)
    }

    pub(crate) async fn to_model_stream<T: DeserializeOwned>(
        &self,
        builder: RequestBuilder,
    ) -> LichessResult<impl Stream<Item = LichessResult<T>>> {
        let stream = self
            .api_call(builder)
            .await?
            .bytes_stream()
            .map_err(|err| Error::new(ErrorKind::Other, err));
        Ok(Box::pin(
            LinesStream::new(StreamReader::new(stream).lines()).filter_map(|line| async move {
                let line = line.ok()?;
                if !line.is_empty() {
                    Some(from_str(&line).map_err(Into::into))
                } else {
                    None
                }
            }),
        ))
    }
}
