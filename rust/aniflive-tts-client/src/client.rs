use crate::error::transport_error;
use crate::stream::{Stop, StreamInner};
use crate::{ApiError, ClientConfig, ClientError, headers, models::*, routes};
use crate::{AudioStream, StreamMetadata};
use reqwest::header::RETRY_AFTER;
use reqwest::{Method, Response};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

pub const MAX_JSON_BYTES: usize = 1024 * 1024;
pub const MAX_ERROR_BYTES: usize = 8192;
pub const DEFAULT_MAX_WAV_BYTES: usize = 64 * 1024 * 1024;

struct Inner {
    http: Mutex<Option<reqwest::Client>>,
    config: ClientConfig,
    stop: Stop,
    streams: Mutex<Vec<Weak<StreamInner>>>,
}
#[derive(Clone)]
pub struct Client {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client").finish_non_exhaustive()
    }
}

impl Client {
    pub fn new(config: ClientConfig) -> Result<Self, ClientError> {
        config.validate()?;
        let http = reqwest::Client::builder()
            .http1_only()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .referer(false)
            .no_gzip()
            .no_brotli()
            .no_deflate()
            .no_zstd()
            .connect_timeout(config.connect_timeout)
            .pool_max_idle_per_host(8)
            .user_agent(format!("aniflive-tts-python-sdk/{}", crate::VERSION))
            .default_headers(config.default_headers.clone())
            .build()
            .map_err(transport_error)?;
        Ok(Self {
            inner: Arc::new(Inner {
                http: Mutex::new(Some(http)),
                config,
                stop: Stop::new(),
                streams: Mutex::new(Vec::new()),
            }),
        })
    }

    /// Close local operations and streams; never invokes remote global cancellation.
    pub fn close(&self) {
        self.inner.stop.close();
        self.inner
            .http
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let mut streams = self.inner.streams.lock().unwrap_or_else(|e| e.into_inner());
        for stream in streams.drain(..).filter_map(|weak| weak.upgrade()) {
            stream.close();
        }
    }
    pub fn is_closed(&self) -> bool {
        self.inner.stop.is_closed()
    }

    pub async fn create_session(
        &self,
        request: &SessionCreate,
    ) -> Result<crate::SpeechSession, ClientError> {
        let body = serde_json::to_value(request).map_err(|_| ClientError::InvalidConfiguration)?;
        let info: SessionInfo = self
            .json_url(
                Method::POST,
                routes::join(self.inner.config.base_url(), "v1/sessions")?,
                Some(body),
                &[201],
            )
            .await?;
        crate::session::validate_info(&info, None)?;
        Ok(crate::SpeechSession::new(self.clone(), info))
    }
    pub async fn get_session(&self, id: &str) -> Result<crate::SpeechSession, ClientError> {
        let info: SessionInfo = self
            .session_json(Method::GET, id, None, None, &[200])
            .await?;
        crate::session::validate_info(&info, Some(id))?;
        Ok(crate::SpeechSession::new(self.clone(), info))
    }
    pub(crate) async fn session_json<T: DeserializeOwned>(
        &self,
        method: Method,
        id: &str,
        suffix: Option<&str>,
        body: Option<Value>,
        statuses: &[u16],
    ) -> Result<T, ClientError> {
        self.json_url(
            method,
            routes::session(self.inner.config.base_url(), id, suffix)?,
            body,
            statuses,
        )
        .await
    }
    pub(crate) async fn session_audio(&self, id: &str) -> Result<AudioStream, ClientError> {
        let response = self
            .response_url(
                Method::GET,
                routes::session(self.inner.config.base_url(), id, Some("audio"))?,
                None,
                &[200],
            )
            .await?;
        if let Some(header) = response.headers().get("x-tts-session-id")
            && header.to_str().ok() != Some(id)
        {
            return Err(ClientError::Protocol("session audio identity mismatch"));
        }
        self.audio_response(response)
    }
    pub async fn stream(&self, request: &SynthesisRequest) -> Result<AudioStream, ClientError> {
        let mut body = synthesis_body(request)?;
        body["stream"] = json!(true);
        body["response_format"] = json!("pcm");
        let response = self
            .response(Method::POST, routes::SPEECH, Some(body))
            .await?;
        self.audio_response(response)
    }

    fn audio_response(&self, response: Response) -> Result<AudioStream, ClientError> {
        let headers = response.headers();
        if headers::content_type(headers)? != "application/octet-stream" {
            return Err(ClientError::Protocol("expected PCM content type"));
        }
        let field = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
        if field("x-tts-stream") != Some("pcm_s16le")
            || field("x-tts-sample-format") != Some("s16le")
        {
            return Err(ClientError::Protocol("invalid PCM format metadata"));
        }
        let channels = field("x-tts-channels")
            .and_then(|v| v.parse::<u16>().ok())
            .filter(|v| *v > 0)
            .ok_or(ClientError::Protocol("invalid PCM channel count"))?;
        let metadata = headers::metadata(headers)?;
        let sample_rate = metadata
            .sample_rate
            .ok_or(ClientError::Protocol("missing PCM sample rate"))?;
        let stream = AudioStream::new(
            response,
            StreamMetadata {
                sample_rate,
                channels,
                sample_format: "s16le".to_owned(),
                metadata,
            },
            self.inner.config.stream_idle_timeout,
            self.inner.config.chunk_bytes,
        );
        let mut streams = self.inner.streams.lock().unwrap_or_else(|e| e.into_inner());
        if self.is_closed() {
            stream.close();
            return Err(ClientError::Closed);
        }
        streams.retain(|weak| weak.strong_count() > 0);
        streams.push(Arc::downgrade(&stream.inner));
        Ok(stream)
    }

    async fn deadline<T>(
        &self,
        started: Instant,
        future: impl std::future::Future<Output = Result<T, ClientError>>,
    ) -> Result<T, ClientError> {
        tokio::select! {
            biased;
            _ = self.inner.stop.cancelled() => Err(ClientError::Closed),
            result = async { match self.inner.config.request_timeout {
            Some(duration) => {
                tokio::time::timeout(duration.saturating_sub(started.elapsed()), future)
                    .await
                    .map_err(|_| ClientError::Timeout)?
            }
            None => future.await,
        }} => result,
        }
    }

    async fn response(
        &self,
        method: Method,
        route: &str,
        body: Option<Value>,
    ) -> Result<Response, ClientError> {
        self.response_url(
            method,
            routes::join(self.inner.config.base_url(), route)?,
            body,
            &[200],
        )
        .await
    }

    async fn response_url(
        &self,
        method: Method,
        url: url::Url,
        body: Option<Value>,
        statuses: &[u16],
    ) -> Result<Response, ClientError> {
        let mut request = {
            let transport = self.inner.http.lock().unwrap_or_else(|e| e.into_inner());
            transport
                .as_ref()
                .ok_or(ClientError::Closed)?
                .request(method, url)
        };
        if let Some(body) = body {
            request = request.json(&body);
        }
        let started = Instant::now();
        let mut response = self
            .deadline(started, async {
                request.send().await.map_err(transport_error)
            })
            .await?;
        if statuses.contains(&response.status().as_u16()) {
            return Ok(response);
        }
        let status_code = response.status().as_u16();
        let retry_after = response
            .headers()
            .get(RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .filter(|v| v.len() <= 256)
            .map(str::to_owned);
        let mut bytes = Vec::new();
        // Error status survives malformed, truncated or oversized diagnostic bodies.
        let diagnostic_timeout = self
            .inner
            .config
            .request_timeout
            .map(|d| d.saturating_sub(started.elapsed()))
            .unwrap_or(Duration::from_secs(5))
            .min(Duration::from_secs(5));
        let diagnostic = tokio::time::timeout(diagnostic_timeout, async {
            while bytes.len() < MAX_ERROR_BYTES {
                match response.chunk().await {
                    Ok(Some(chunk)) => {
                        let take = chunk.len().min(MAX_ERROR_BYTES - bytes.len());
                        bytes.extend_from_slice(&chunk[..take]);
                        if take < chunk.len() {
                            break;
                        }
                    }
                    _ => break,
                }
            }
        });
        tokio::select! {
            biased;
            _ = self.inner.stop.cancelled() => return Err(ClientError::Closed),
            _ = diagnostic => {}
        }
        let value: Option<Value> = serde_json::from_slice(&bytes).ok();
        let server_code = value
            .as_ref()
            .and_then(|v| v.get("code"))
            .cloned()
            .filter(|v| v.is_number() || v.as_str().is_some_and(|s| s.len() <= 128));
        let message = value
            .as_ref()
            .and_then(|v| v.get("message"))
            .and_then(Value::as_str)
            .map(|s| s.chars().take(2048).collect())
            .unwrap_or_else(|| "service error".to_owned());
        Err(ClientError::Api(ApiError {
            status_code,
            server_code,
            message,
            retry_after,
            diagnostic: (!bytes.is_empty()).then(|| String::from_utf8_lossy(&bytes).into_owned()),
        }))
    }

    async fn json<T: DeserializeOwned>(
        &self,
        method: Method,
        route: &str,
        body: Option<Value>,
    ) -> Result<T, ClientError> {
        self.json_url(
            method,
            routes::join(self.inner.config.base_url(), route)?,
            body,
            &[200],
        )
        .await
    }

    async fn json_url<T: DeserializeOwned>(
        &self,
        method: Method,
        url: url::Url,
        body: Option<Value>,
        statuses: &[u16],
    ) -> Result<T, ClientError> {
        let started = Instant::now();
        let response = self.response_url(method, url, body, statuses).await?;
        self.deadline(started, async {
            let media = headers::content_type(response.headers())?;
            if media != "application/json" {
                return Err(ClientError::Protocol("expected JSON content type"));
            }
            let bytes = read_bounded(response, MAX_JSON_BYTES).await?;
            serde_json::from_slice(&bytes).map_err(|_| ClientError::Protocol("invalid JSON schema"))
        })
        .await
    }

    pub async fn health(&self) -> Result<Health, ClientError> {
        self.json(Method::GET, routes::HEALTH, None).await
    }
    pub async fn capabilities(&self) -> Result<Capabilities, ClientError> {
        self.json(Method::GET, routes::CAPABILITIES, None).await
    }
    pub async fn models(&self) -> Result<ListResponse<ModelInfo>, ClientError> {
        self.json(Method::GET, routes::MODELS, None).await
    }
    pub async fn voices(&self) -> Result<ListResponse<VoiceInfo>, ClientError> {
        self.json(Method::GET, routes::VOICES, None).await
    }
    pub async fn expressions(&self) -> Result<ExpressionCatalogInfo, ClientError> {
        self.json(Method::GET, routes::EXPRESSIONS, None).await
    }
    pub async fn activate_model(&self, model: &str) -> Result<ActivationResult, ClientError> {
        self.json(
            Method::POST,
            routes::ACTIVATE,
            Some(json!({"model": model})),
        )
        .await
    }
    /// Global current-owner cancellation, not scoped to this client's requests.
    pub async fn cancel(&self) -> Result<CancellationResult, ClientError> {
        self.json(Method::POST, routes::CANCEL, None).await
    }
    pub async fn synthesize(
        &self,
        request: &SynthesisRequest,
    ) -> Result<SynthesisResult, ClientError> {
        let mut body = synthesis_body(request)?;
        body["stream"] = json!(false);
        body["response_format"] = json!("wav");
        let started = Instant::now();
        let response = self
            .response(Method::POST, routes::SPEECH, Some(body))
            .await?;
        self.deadline(started, async {
            let content_type = headers::content_type(response.headers())?;
            if content_type != "audio/wav" {
                return Err(ClientError::Protocol("expected WAV content type"));
            }
            let metadata = headers::metadata(response.headers())?;
            let audio = read_bounded(response, self.inner.config.max_wav_bytes).await?;
            if audio.len() < 12 || &audio[..4] != b"RIFF" || &audio[8..12] != b"WAVE" {
                return Err(ClientError::Protocol("invalid WAV signature"));
            }
            Ok(SynthesisResult {
                audio,
                content_type,
                metadata,
            })
        })
        .await
    }
    pub async fn probe(&self, required: &[RequiredFeature]) -> Result<ServiceInfo, ClientError> {
        let health = self.health().await?;
        let capabilities = self.capabilities().await?;
        if health.service != "AnifLive-TTS" || capabilities.service != "AnifLive-TTS" {
            return Err(ClientError::Compatibility("service identity mismatch"));
        }
        if health.version.is_empty()
            || capabilities.version.is_empty()
            || capabilities.languages.is_empty()
        {
            return Err(ClientError::Compatibility(
                "missing service contract metadata",
            ));
        }
        for feature in required {
            let supported = match feature {
                RequiredFeature::PcmStreaming => capabilities.streaming.pcm16,
                // Protocol support does not imply the active package has profiles.
                RequiredFeature::Expressions => capabilities.expression.native,
                RequiredFeature::SpeechSessions => capabilities
                    .speech_sessions
                    .as_ref()
                    .is_some_and(|s| s.committed_segments && s.pcm_stream),
            };
            if !supported {
                return Err(ClientError::Compatibility("required feature unavailable"));
            }
        }
        // ready=false describes runtime readiness, not protocol incompatibility.
        Ok(ServiceInfo {
            health,
            capabilities,
        })
    }
}

pub(crate) fn synthesis_body(request: &SynthesisRequest) -> Result<Value, ClientError> {
    if request.text.is_some() == request.segments.is_some() {
        return Err(ClientError::InvalidConfiguration);
    }
    // Reject nonfinite floats instead of silently serializing them as JSON null.
    if let Some(g) = &request.generation
        && [g.top_p, g.temperature, g.noise_scale, g.speed]
            .into_iter()
            .flatten()
            .any(|v| !v.is_finite())
    {
        return Err(ClientError::InvalidConfiguration);
    }
    let expressions = request.expression.iter().chain(
        request
            .segments
            .iter()
            .flatten()
            .filter_map(|s| s.expression.as_ref()),
    );
    if expressions
        .into_iter()
        .filter_map(|e| e.intensity)
        .any(|v| !v.is_finite())
    {
        return Err(ClientError::InvalidConfiguration);
    }
    serde_json::to_value(request).map_err(|_| ClientError::InvalidConfiguration)
}
async fn read_bounded(mut response: Response, maximum: usize) -> Result<Vec<u8>, ClientError> {
    if response
        .content_length()
        .is_some_and(|n| n > maximum as u64)
    {
        return Err(ClientError::Protocol("response body exceeds limit"));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(transport_error)? {
        if chunk.len() > maximum.saturating_sub(bytes.len()) {
            return Err(ClientError::Protocol("response body exceeds limit"));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}
