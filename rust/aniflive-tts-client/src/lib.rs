mod client;
mod client_config;
mod error;
mod headers;
pub mod models;
mod routes;
mod stream;
pub use stream::{AudioStream, MAX_STREAM_BUFFER_BYTES, StreamMetadata};

pub use client::{Client, DEFAULT_MAX_WAV_BYTES, MAX_ERROR_BYTES, MAX_JSON_BYTES};
pub use client_config::ClientConfig;
pub use error::{ApiError, ClientError};
pub use models::*;
pub use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
/// PEP 440 version shared by the Python module and HTTP User-Agent.
/// Cargo uses SemVer pre-release syntax; keep the mapping checked below.
pub const VERSION: &str = "1.5.0b1";

#[cfg(test)]
mod version_tests {
    #[test]
    fn cargo_semver_matches_public_pep440_version() {
        assert_eq!(
            env!("CARGO_PKG_VERSION").replace("-beta.", "b"),
            super::VERSION
        );
    }
}

mod session;
pub use session::SpeechSession;
