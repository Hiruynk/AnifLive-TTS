use serde_json::Value;
use std::fmt;
use thiserror::Error;
use url::ParseError;

/// The server message/diagnostic are explicitly inspected attributes, not log-safe text.
#[derive(Clone)]
pub struct ApiError {
    pub status_code: u16,
    pub server_code: Option<Value>,
    pub message: String,
    pub retry_after: Option<String>,
    pub diagnostic: Option<String>,
}
impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "service returned HTTP {}", self.status_code)
    }
}
impl fmt::Debug for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl std::error::Error for ApiError {}

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("invalid base URL")]
    InvalidBaseUrl(#[source] ParseError),
    #[error("base URL scheme must be http or https")]
    UnsupportedScheme,
    #[error("base URL must not contain credentials")]
    CredentialsNotAllowed,
    #[error("base URL must not contain a query string")]
    QueryNotAllowed,
    #[error("base URL contains a fragment")]
    FragmentNotAllowed,
    #[error("invalid client configuration")]
    InvalidConfiguration,
    #[error("client is closed")]
    Closed,
    #[error("another stream read is pending")]
    ConcurrentRead,
    #[error("transport request failed")]
    Transport,
    #[error("request timed out")]
    Timeout,
    #[error("invalid service response: {0}")]
    Protocol(&'static str),
    #[error("incompatible service: {0}")]
    Compatibility(&'static str),
    #[error("{0}")]
    Api(ApiError),
}
impl ClientError {
    /// Stable binding classification without parsing an exception message.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Closed => "ClosedError",
            Self::ConcurrentRead => "ConcurrentReadError",
            Self::Transport => "TransportError",
            Self::Timeout => "TimeoutError",
            Self::Protocol(_) => "ProtocolError",
            Self::Compatibility(_) => "CompatibilityError",
            Self::Api(error) => match error.status_code {
                400 => "BadRequestError",
                404 => "NotFoundError",
                409 => "ConflictError",
                413 => "PayloadTooLargeError",
                429 => "BusyError",
                500..=599 => "ServerError",
                _ => "APIError",
            },
            _ => "ClientError",
        }
    }
}
pub(crate) fn transport_error(error: reqwest::Error) -> ClientError {
    // Never retain the reqwest error: its Display/Debug/source can contain URLs.
    if error.is_timeout() {
        ClientError::Timeout
    } else {
        ClientError::Transport
    }
}
