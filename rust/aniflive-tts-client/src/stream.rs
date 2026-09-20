use crate::error::transport_error;
use crate::{ClientError, SynthesisMetadata};
use reqwest::Response;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, watch};

pub const MAX_STREAM_BUFFER_BYTES: usize = 1024 * 1024;
#[derive(Clone)]
pub(crate) struct Stop(watch::Sender<bool>);
impl Stop {
    pub(crate) fn new() -> Self {
        Self(watch::channel(false).0)
    }
    pub(crate) fn close(&self) {
        self.0.send_replace(true);
    }
    pub(crate) fn is_closed(&self) -> bool {
        *self.0.borrow()
    }
    pub(crate) async fn cancelled(&self) {
        let mut receiver = self.0.subscribe();
        loop {
            if *receiver.borrow_and_update() {
                return;
            }
            if receiver.changed().await.is_err() {
                return;
            }
        }
    }
}
#[derive(Debug, Clone)]
pub struct StreamMetadata {
    pub sample_rate: u32,
    pub channels: u16,
    pub sample_format: String,
    pub metadata: SynthesisMetadata,
}
struct State {
    response: Option<Response>,
    pending: Vec<u8>,
    offset: usize,
    total: u64,
}
pub(crate) struct StreamInner {
    stop: Stop,
    state: Mutex<State>,
    metadata: StreamMetadata,
    idle_timeout: Option<Duration>,
    chunk_bytes: usize,
}
impl StreamInner {
    pub(crate) fn close(&self) {
        self.stop.close();
        if let Ok(mut state) = self.state.try_lock() {
            state.response = None;
            state.pending.clear();
        }
    }
}
/// Pull-driven PCM bytes. Chunk boundaries need not align with sample frames.
#[derive(Clone)]
pub struct AudioStream {
    pub(crate) inner: Arc<StreamInner>,
}
impl std::fmt::Debug for AudioStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioStream")
            .field("closed", &self.is_closed())
            .finish_non_exhaustive()
    }
}
// A cancelled/dropped next_chunk future must release the response too.
struct ReadGuard<'a> {
    state: tokio::sync::MutexGuard<'a, State>,
    stop: &'a Stop,
    completed: bool,
}
impl Drop for ReadGuard<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.stop.close();
            self.state.response = None;
            self.state.pending.clear();
        }
    }
}
impl AudioStream {
    pub(crate) fn new(
        response: Response,
        metadata: StreamMetadata,
        idle_timeout: Option<Duration>,
        chunk_bytes: usize,
    ) -> Self {
        Self {
            inner: Arc::new(StreamInner {
                stop: Stop::new(),
                state: Mutex::new(State {
                    response: Some(response),
                    pending: Vec::new(),
                    offset: 0,
                    total: 0,
                }),
                metadata,
                idle_timeout,
                chunk_bytes,
            }),
        }
    }
    pub fn metadata(&self) -> &StreamMetadata {
        &self.inner.metadata
    }
    pub fn sample_rate(&self) -> u32 {
        self.inner.metadata.sample_rate
    }
    pub fn channels(&self) -> u16 {
        self.inner.metadata.channels
    }
    pub fn close(&self) {
        self.inner.close();
    }
    pub fn is_closed(&self) -> bool {
        self.inner.stop.is_closed()
    }
    pub async fn next_chunk(&self) -> Result<Option<Vec<u8>>, ClientError> {
        if self.is_closed() {
            return Ok(None);
        }
        let state = self
            .inner
            .state
            .try_lock()
            .map_err(|_| ClientError::ConcurrentRead)?;
        let mut guard = ReadGuard {
            state,
            stop: &self.inner.stop,
            completed: false,
        };
        loop {
            if self.is_closed() {
                return Ok(None);
            }
            if guard.state.offset < guard.state.pending.len() {
                let end =
                    (guard.state.offset + self.inner.chunk_bytes).min(guard.state.pending.len());
                let result = guard.state.pending[guard.state.offset..end].to_vec();
                guard.state.offset = end;
                if end == guard.state.pending.len() {
                    guard.state.pending.clear();
                    guard.state.offset = 0;
                }
                guard.completed = true;
                return Ok(Some(result));
            }
            let Some(response) = guard.state.response.as_mut() else {
                return Ok(None);
            };
            let read = async {
                match self.inner.idle_timeout {
                    Some(timeout) => tokio::time::timeout(timeout, response.chunk())
                        .await
                        .map_err(|_| ClientError::Timeout)?
                        .map_err(transport_error),
                    None => response.chunk().await.map_err(transport_error),
                }
            };
            let chunk = tokio::select! {
                biased;
                _ = self.inner.stop.cancelled() => return Ok(None),
                result = read => result?,
            };
            match chunk {
                Some(chunk) => {
                    if chunk.len() > MAX_STREAM_BUFFER_BYTES {
                        return Err(ClientError::Protocol(
                            "stream transport chunk exceeds bounded buffer limit",
                        ));
                    }
                    guard.state.total = guard
                        .state
                        .total
                        .checked_add(chunk.len() as u64)
                        .ok_or(ClientError::Protocol("PCM byte count overflow"))?;
                    guard.state.pending = chunk.to_vec();
                    guard.state.offset = 0;
                }
                None => {
                    if !guard
                        .state
                        .total
                        .is_multiple_of(2 * u64::from(self.channels()))
                    {
                        return Err(ClientError::Protocol("truncated PCM sample frame"));
                    }
                    return Ok(None);
                }
            }
        }
    }
}
