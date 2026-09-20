use crate::client::synthesis_body;
use crate::stream::StreamInner;
use crate::{AudioStream, Client, ClientError, SegmentInfo, SegmentRequest, SessionInfo};
use reqwest::Method;
use std::sync::{Arc, Mutex, Weak};

struct SessionInner {
    client: Client,
    initial: SessionInfo,
    streams: Mutex<Vec<Weak<StreamInner>>>,
}
impl SessionInner {
    fn close_readers(&self) {
        let mut streams = self.streams.lock().unwrap_or_else(|e| e.into_inner());
        for stream in streams.drain(..).filter_map(|weak| weak.upgrade()) {
            stream.close();
        }
    }
}
impl Drop for SessionInner {
    fn drop(&mut self) {
        self.close_readers();
    }
}
/// Remote session handle. Its initial information is a snapshot, never a state cache.
#[derive(Clone)]
pub struct SpeechSession {
    inner: Arc<SessionInner>,
}
impl std::fmt::Debug for SpeechSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeechSession").finish_non_exhaustive()
    }
}
impl SpeechSession {
    pub(crate) fn new(client: Client, initial: SessionInfo) -> Self {
        Self {
            inner: Arc::new(SessionInner {
                client,
                initial,
                streams: Mutex::new(Vec::new()),
            }),
        }
    }
    pub fn id(&self) -> &str {
        &self.inner.initial.id
    }
    pub fn initial_info(&self) -> &SessionInfo {
        &self.inner.initial
    }
    async fn operation(
        &self,
        method: Method,
        suffix: Option<&str>,
    ) -> Result<SessionInfo, ClientError> {
        let info: SessionInfo = self
            .inner
            .client
            .session_json(method, self.id(), suffix, None, &[200])
            .await?;
        validate_info(&info, Some(self.id()))?;
        Ok(info)
    }
    pub async fn get(&self) -> Result<SessionInfo, ClientError> {
        self.operation(Method::GET, None).await
    }
    pub async fn flush(&self) -> Result<SessionInfo, ClientError> {
        self.operation(Method::POST, Some("flush")).await
    }
    pub async fn cancel(&self) -> Result<SessionInfo, ClientError> {
        self.inner.close_readers();
        self.operation(Method::POST, Some("cancel")).await
    }
    /// Explicit remote DELETE is the server's cancel alias; it retains the terminal record.
    pub async fn close(&self) -> Result<SessionInfo, ClientError> {
        self.inner.close_readers();
        self.operation(Method::DELETE, None).await
    }
    pub async fn append(&self, request: &SegmentRequest) -> Result<SegmentInfo, ClientError> {
        let mut synthesis = crate::SynthesisRequest::new(&request.text, request.language);
        synthesis.expression = request.expression.clone();
        synthesis.generation = request.generation.clone();
        synthesis_body(&synthesis)?; // finite JSON numbers; do not enforce backend-specific ranges.
        let body = serde_json::to_value(request).map_err(|_| ClientError::InvalidConfiguration)?;
        let info: SegmentInfo = self
            .inner
            .client
            .session_json(
                Method::POST,
                self.id(),
                Some("segments"),
                Some(body),
                &[200, 201],
            )
            .await?;
        if info.segment_id != request.segment_id {
            return Err(ClientError::Protocol("session segment identity mismatch"));
        }
        Ok(info)
    }
    pub async fn audio(&self) -> Result<AudioStream, ClientError> {
        let stream = self.inner.client.session_audio(self.id()).await?;
        let mut streams = self.inner.streams.lock().unwrap_or_else(|e| e.into_inner());
        streams.retain(|weak| weak.strong_count() > 0);
        streams.push(Arc::downgrade(&stream.inner));
        Ok(stream)
    }
}
pub(crate) fn validate_info(info: &SessionInfo, expected: Option<&str>) -> Result<(), ClientError> {
    if info.id.is_empty()
        || info.id == "."
        || info.id == ".."
        || info.object != "speech.session"
        || info.sample_rate == 0
        || info.format != "pcm_s16le"
    {
        return Err(ClientError::Protocol("invalid session metadata"));
    }
    if expected.is_some_and(|id| info.id != id) {
        return Err(ClientError::Protocol("session identity mismatch"));
    }
    Ok(())
}
