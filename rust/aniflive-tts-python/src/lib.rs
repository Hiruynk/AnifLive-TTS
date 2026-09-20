mod bridge;
mod convert;
use aniflive_tts_client as core;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;
use std::time::Duration;

fn duration(value: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(value).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("timeout must be finite and non-negative")
    })
}
fn stream_object(
    py: Python<'_>,
    inner: core::AudioStream,
    affinity: bridge::Affinity,
) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, NativeAudioStream { inner, affinity })?.into_any())
}
fn session_object(
    py: Python<'_>,
    inner: core::SpeechSession,
    affinity: bridge::Affinity,
) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, NativeSpeechSession { inner, affinity })?.into_any())
}
#[pyclass(module = "aniflive_tts._native", frozen)]
struct NativeClient {
    inner: core::Client,
    affinity: bridge::Affinity,
}
#[pymethods]
impl NativeClient {
    #[new]
    #[pyo3(signature=(base_url, *, connect_timeout=10., request_timeout=Some(120.), stream_idle_timeout=Some(60.), headers=None, max_wav_bytes=67108864, chunk_bytes=16384))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        base_url: &str,
        connect_timeout: f64,
        request_timeout: Option<f64>,
        stream_idle_timeout: Option<f64>,
        headers: Option<BTreeMap<String, String>>,
        max_wav_bytes: usize,
        chunk_bytes: usize,
    ) -> PyResult<Self> {
        let mut defaults = core::HeaderMap::new();
        for (name, value) in headers.unwrap_or_default() {
            let name = core::HeaderName::from_bytes(name.as_bytes())
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid header name"))?;
            let value = core::HeaderValue::from_str(&value)
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid header value"))?;
            defaults.insert(name, value);
        }
        let config = core::ClientConfig::new(base_url)
            .map_err(|e| convert::error(py, e))?
            .with_connect_timeout(duration(connect_timeout)?)
            .with_request_timeout(request_timeout.map(duration).transpose()?)
            .with_stream_idle_timeout(stream_idle_timeout.map(duration).transpose()?)
            .with_default_headers(defaults)
            .with_max_wav_bytes(max_wav_bytes)
            .with_chunk_bytes(chunk_bytes);
        Ok(Self {
            inner: core::Client::new(config).map_err(|e| convert::error(py, e))?,
            affinity: bridge::affinity(),
        })
    }
    fn close(&self) {
        self.inner.close();
    }
    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
    fn health(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::sync(py, async move { inner.health().await }, |py, value| {
            convert::serialized(py, "Health", value)
        })
    }
    fn health_async(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.health().await },
            |py, value| convert::serialized(py, "Health", value),
        )
    }
    fn capabilities(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::sync(
            py,
            async move { inner.capabilities().await },
            |py, value| convert::serialized(py, "Capabilities", value),
        )
    }
    fn capabilities_async(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.capabilities().await },
            |py, value| convert::serialized(py, "Capabilities", value),
        )
    }
    fn models(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::sync(py, async move { inner.models().await }, |py, value| {
            convert::serialized(py, "Models", value)
        })
    }
    fn models_async(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.models().await },
            |py, value| convert::serialized(py, "Models", value),
        )
    }
    fn voices(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::sync(py, async move { inner.voices().await }, |py, value| {
            convert::serialized(py, "Voices", value)
        })
    }
    fn voices_async(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.voices().await },
            |py, value| convert::serialized(py, "Voices", value),
        )
    }
    fn expressions(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::sync(py, async move { inner.expressions().await }, |py, value| {
            convert::serialized(py, "ExpressionCatalogInfo", value)
        })
    }
    fn expressions_async(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.expressions().await },
            |py, value| convert::serialized(py, "ExpressionCatalogInfo", value),
        )
    }
    fn activate_model(&self, py: Python<'_>, model: String) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::sync(
            py,
            async move { inner.activate_model(&model).await },
            |py, value| convert::serialized(py, "ActivationResult", value),
        )
    }
    fn activate_model_async(&self, py: Python<'_>, model: String) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.activate_model(&model).await },
            |py, value| convert::serialized(py, "ActivationResult", value),
        )
    }
    fn cancel(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::sync(py, async move { inner.cancel().await }, |py, value| {
            convert::serialized(py, "CancellationResult", value)
        })
    }
    fn cancel_async(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.cancel().await },
            |py, value| convert::serialized(py, "CancellationResult", value),
        )
    }
    #[pyo3(signature=(required_features=Vec::new()))]
    fn probe(&self, py: Python<'_>, required_features: Vec<String>) -> PyResult<Py<PyAny>> {
        let required_features = required_features
            .iter()
            .map(|value| match value.as_str() {
                "pcm_streaming" => Ok(core::RequiredFeature::PcmStreaming),
                "expressions" => Ok(core::RequiredFeature::Expressions),
                "speech_sessions" => Ok(core::RequiredFeature::SpeechSessions),
                _ => Err(pyo3::exceptions::PyValueError::new_err(
                    "unknown required feature",
                )),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let inner = self.inner.clone();
        bridge::sync(
            py,
            async move { inner.probe(&required_features).await },
            |py, value| convert::serialized(py, "ServiceInfo", value),
        )
    }
    #[pyo3(signature=(required_features=Vec::new()))]
    fn probe_async(&self, py: Python<'_>, required_features: Vec<String>) -> PyResult<Py<PyAny>> {
        let required_features = required_features
            .iter()
            .map(|value| match value.as_str() {
                "pcm_streaming" => Ok(core::RequiredFeature::PcmStreaming),
                "expressions" => Ok(core::RequiredFeature::Expressions),
                "speech_sessions" => Ok(core::RequiredFeature::SpeechSessions),
                _ => Err(pyo3::exceptions::PyValueError::new_err(
                    "unknown required feature",
                )),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let inner = self.inner.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.probe(&required_features).await },
            |py, value| convert::serialized(py, "ServiceInfo", value),
        )
    }
    fn synthesize(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let request: core::SynthesisRequest = convert::request(request, "SynthesisRequest")?;
        let inner = self.inner.clone();
        bridge::sync(
            py,
            async move { inner.synthesize(&request).await },
            convert::wav,
        )
    }
    fn synthesize_async(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let request: core::SynthesisRequest = convert::request(request, "SynthesisRequest")?;
        let inner = self.inner.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.synthesize(&request).await },
            convert::wav,
        )
    }
    fn stream(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let request: core::SynthesisRequest = convert::request(request, "SynthesisRequest")?;
        let inner = self.inner.clone();
        let affinity = self.affinity.clone();
        bridge::sync(
            py,
            async move { inner.stream(&request).await },
            move |py, value| stream_object(py, value, affinity),
        )
    }
    fn stream_async(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let request: core::SynthesisRequest = convert::request(request, "SynthesisRequest")?;
        let inner = self.inner.clone();
        let affinity = self.affinity.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.stream(&request).await },
            move |py, value| stream_object(py, value, affinity),
        )
    }
    fn create_session(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let request: core::SessionCreate = convert::request(request, "SessionCreate")?;
        let inner = self.inner.clone();
        let affinity = self.affinity.clone();
        bridge::sync(
            py,
            async move { inner.create_session(&request).await },
            move |py, value| session_object(py, value, affinity),
        )
    }
    fn create_session_async(
        &self,
        py: Python<'_>,
        request: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let request: core::SessionCreate = convert::request(request, "SessionCreate")?;
        let inner = self.inner.clone();
        let affinity = self.affinity.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.create_session(&request).await },
            move |py, value| session_object(py, value, affinity),
        )
    }
    fn get_session(&self, py: Python<'_>, id: String) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        let affinity = self.affinity.clone();
        bridge::sync(
            py,
            async move { inner.get_session(&id).await },
            move |py, value| session_object(py, value, affinity),
        )
    }
    fn get_session_async(&self, py: Python<'_>, id: String) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        let affinity = self.affinity.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.get_session(&id).await },
            move |py, value| session_object(py, value, affinity),
        )
    }
}

#[pyclass(module = "aniflive_tts._native", frozen)]
struct NativeAudioStream {
    inner: core::AudioStream,
    affinity: bridge::Affinity,
}
#[pymethods]
impl NativeAudioStream {
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }
    #[getter]
    fn channels(&self) -> u16 {
        self.inner.channels()
    }
    #[getter]
    fn sample_format(&self) -> &str {
        &self.inner.metadata().sample_format
    }
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let metadata = self.inner.metadata();
        let kwargs = PyDict::new(py);
        kwargs.set_item("sample_rate", metadata.sample_rate)?;
        kwargs.set_item("channels", metadata.channels)?;
        kwargs.set_item("sample_format", &metadata.sample_format)?;
        kwargs.set_item(
            "metadata",
            convert::serialized(py, "SynthesisMetadata", &metadata.metadata)?,
        )?;
        Ok(py
            .import("aniflive_tts.sdk.types")?
            .getattr("StreamMetadata")?
            .call((), Some(&kwargs))?
            .unbind())
    }
    fn close(&self) {
        self.inner.close();
    }
    fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
    fn next_chunk(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::sync(py, async move { inner.next_chunk().await }, convert::chunk)
    }
    fn next_chunk_async(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.next_chunk().await },
            convert::chunk,
        )
    }
}

#[pyclass(module = "aniflive_tts._native", frozen)]
struct NativeSpeechSession {
    inner: core::SpeechSession,
    affinity: bridge::Affinity,
}
#[pymethods]
impl NativeSpeechSession {
    #[getter]
    fn id(&self) -> &str {
        self.inner.id()
    }
    #[getter]
    fn initial_info(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        convert::serialized(py, "SessionInfo", self.inner.initial_info())
    }
    fn get(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::sync(py, async move { inner.get().await }, |py, value| {
            convert::serialized(py, "SessionInfo", value)
        })
    }
    fn get_async(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.get().await },
            |py, value| convert::serialized(py, "SessionInfo", value),
        )
    }
    fn flush(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::sync(py, async move { inner.flush().await }, |py, value| {
            convert::serialized(py, "SessionInfo", value)
        })
    }
    fn flush_async(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.flush().await },
            |py, value| convert::serialized(py, "SessionInfo", value),
        )
    }
    fn cancel(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::sync(py, async move { inner.cancel().await }, |py, value| {
            convert::serialized(py, "SessionInfo", value)
        })
    }
    fn cancel_async(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.cancel().await },
            |py, value| convert::serialized(py, "SessionInfo", value),
        )
    }
    fn close(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::sync(py, async move { inner.close().await }, |py, value| {
            convert::serialized(py, "SessionInfo", value)
        })
    }
    fn close_async(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.close().await },
            |py, value| convert::serialized(py, "SessionInfo", value),
        )
    }
    fn append(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let request: core::SegmentRequest = convert::request(request, "SegmentRequest")?;
        let inner = self.inner.clone();
        bridge::sync(
            py,
            async move { inner.append(&request).await },
            |py, value| convert::serialized(py, "SegmentInfo", value),
        )
    }
    fn append_async(&self, py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let request: core::SegmentRequest = convert::request(request, "SegmentRequest")?;
        let inner = self.inner.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.append(&request).await },
            |py, value| convert::serialized(py, "SegmentInfo", value),
        )
    }
    fn audio(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        let affinity = self.affinity.clone();
        bridge::sync(py, async move { inner.audio().await }, move |py, value| {
            stream_object(py, value, affinity)
        })
    }
    fn audio_async(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.inner.clone();
        let affinity = self.affinity.clone();
        bridge::asynchronous(
            py,
            &self.affinity,
            async move { inner.audio().await },
            move |py, value| stream_object(py, value, affinity),
        )
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", core::VERSION)?;
    module.add_class::<NativeClient>()?;
    module.add_class::<NativeAudioStream>()?;
    module.add_class::<NativeSpeechSession>()?;
    Ok(())
}
