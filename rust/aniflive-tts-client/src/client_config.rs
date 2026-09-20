use reqwest::header::HeaderMap;
use std::time::Duration;
use url::Url;

use crate::ClientError;

/// Local configuration for an AnifLive-TTS client.
///
/// This type only validates and normalizes the service base URL. It does not
/// perform network requests.
#[derive(Clone)]
pub struct ClientConfig {
    base_url: Url,
    pub(crate) connect_timeout: Duration,
    pub(crate) request_timeout: Option<Duration>,
    pub(crate) stream_idle_timeout: Option<Duration>,
    pub(crate) default_headers: HeaderMap,
    pub(crate) chunk_bytes: usize,
    pub(crate) max_wav_bytes: usize,
}

impl ClientConfig {
    /// Creates a configuration for an absolute HTTP or HTTPS base URL.
    ///
    /// The URL must not contain user information, a query, or a fragment.
    /// A single trailing slash is added to the path so callers can later join
    /// endpoint paths without dropping a configured path prefix.
    pub fn new(base_url: impl AsRef<str>) -> Result<Self, ClientError> {
        let mut base_url = Url::parse(base_url.as_ref()).map_err(ClientError::InvalidBaseUrl)?;

        if !matches!(base_url.scheme(), "http" | "https") {
            return Err(ClientError::UnsupportedScheme);
        }

        if !base_url.username().is_empty() || base_url.password().is_some() {
            return Err(ClientError::CredentialsNotAllowed);
        }

        if base_url.query().is_some() {
            return Err(ClientError::QueryNotAllowed);
        }

        if base_url.fragment().is_some() {
            return Err(ClientError::FragmentNotAllowed);
        }

        let trailing_slashes = base_url
            .path()
            .bytes()
            .rev()
            .take_while(|byte| *byte == b'/')
            .count();
        {
            let mut path_segments = base_url
                .path_segments_mut()
                .expect("HTTP and HTTPS URLs have hierarchical paths");
            for _ in 0..trailing_slashes {
                path_segments.pop_if_empty();
            }
            path_segments.push("");
        }

        Ok(Self {
            base_url,
            connect_timeout: Duration::from_secs(10),
            request_timeout: Some(Duration::from_secs(120)),
            stream_idle_timeout: Some(Duration::from_secs(60)),
            default_headers: HeaderMap::new(),
            chunk_bytes: 16384,
            max_wav_bytes: crate::DEFAULT_MAX_WAV_BYTES,
        })
    }

    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }
    pub fn with_request_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.request_timeout = timeout;
        self
    }
    pub fn with_stream_idle_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.stream_idle_timeout = timeout;
        self
    }
    pub fn with_default_headers(mut self, headers: HeaderMap) -> Self {
        self.default_headers = headers;
        self
    }
    pub fn with_chunk_bytes(mut self, size: usize) -> Self {
        self.chunk_bytes = size;
        self
    }
    pub fn with_max_wav_bytes(mut self, maximum: usize) -> Self {
        self.max_wav_bytes = maximum;
        self
    }
    pub(crate) fn validate(&self) -> Result<(), ClientError> {
        if self.connect_timeout.is_zero()
            || self.request_timeout.is_some_and(|v| v.is_zero())
            || self.stream_idle_timeout.is_some_and(|v| v.is_zero())
            || self.max_wav_bytes == 0
            || !(1..=65536).contains(&self.chunk_bytes)
        {
            return Err(ClientError::InvalidConfiguration);
        }
        Ok(())
    }

    /// Returns the normalized service base URL.
    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    /// Returns the normalized service base URL as a string.
    pub fn base_url_str(&self) -> &str {
        self.base_url.as_str()
    }
}

impl std::fmt::Debug for ClientConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientConfig").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_trailing_slashes_and_preserves_path_prefix() {
        let config = ClientConfig::new("http://127.0.0.1:9880/service/proxy///").unwrap();

        assert_eq!(
            config.base_url_str(),
            "http://127.0.0.1:9880/service/proxy/"
        );
    }

    #[test]
    fn normalizes_root_url() {
        let config = ClientConfig::new("https://example.com").unwrap();

        assert_eq!(config.base_url_str(), "https://example.com/");
    }

    #[test]
    fn preserves_percent_encoded_path_prefix() {
        let config = ClientConfig::new("https://example.com/proxy%2Fedge///").unwrap();

        assert_eq!(config.base_url_str(), "https://example.com/proxy%2Fedge/");
    }

    #[test]
    fn rejects_non_http_schemes() {
        let error = ClientConfig::new("ftp://example.com/").unwrap_err();

        assert!(matches!(error, ClientError::UnsupportedScheme));
    }

    #[test]
    fn rejects_credentials_without_echoing_them() {
        let error = ClientConfig::new("https://alice:secret@example.com/proxy/").unwrap_err();

        assert!(matches!(error, ClientError::CredentialsNotAllowed));
        assert!(!format!("{error:?}").contains("alice"));
        assert!(!format!("{error:?}").contains("secret"));
    }

    #[test]
    fn rejects_queries_without_echoing_them() {
        let error = ClientConfig::new("https://example.com/proxy?token=secret").unwrap_err();

        assert!(matches!(error, ClientError::QueryNotAllowed));
        assert!(!format!("{error:?}").contains("secret"));
    }

    #[test]
    fn rejects_fragments() {
        let error = ClientConfig::new("https://example.com/proxy#private").unwrap_err();

        assert!(matches!(error, ClientError::FragmentNotAllowed));
        assert!(!format!("{error:?}").contains("private"));
    }

    #[test]
    fn malformed_urls_use_a_redacted_error() {
        let input = "not a URL with private credentials";
        let error = ClientConfig::new(input).unwrap_err();

        assert!(matches!(error, ClientError::InvalidBaseUrl(_)));
        assert!(!format!("{error:?}").contains("private"));
        assert!(!error.to_string().contains(input));
    }
}
