use crate::{ClientError, SynthesisMetadata};
use reqwest::header::{CONTENT_TYPE, HeaderMap};
use std::collections::BTreeMap;

pub(crate) fn content_type(headers: &HeaderMap) -> Result<String, ClientError> {
    headers
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .map(|v| v.trim().to_ascii_lowercase())
        .ok_or(ClientError::Protocol("missing content type"))
}
pub(crate) fn metadata(headers: &HeaderMap) -> Result<SynthesisMetadata, ClientError> {
    let mut safe = BTreeMap::new();
    for (name, value) in headers {
        if name.as_str().starts_with("x-tts-")
            || name.as_str().starts_with("x-tensorrt-")
            || name.as_str() == "x-pytorch-fallback"
        {
            let value = value
                .to_str()
                .map_err(|_| ClientError::Protocol("invalid metadata header"))?;
            if value.len() > 4096 {
                return Err(ClientError::Protocol("metadata header too large"));
            }
            safe.insert(name.to_string(), value.to_owned());
        }
    }
    let sample_rate = safe
        .get("x-tts-sample-rate")
        .map(|v| {
            v.parse::<u32>()
                .ok()
                .filter(|v| *v > 0)
                .ok_or(ClientError::Protocol("invalid sample rate"))
        })
        .transpose()?;
    Ok(SynthesisMetadata {
        sample_rate,
        model: safe.get("x-tts-model").cloned(),
        service_version: safe.get("x-tts-version").cloned(),
        backend: safe
            .get("x-tts-backend")
            .or_else(|| safe.get("x-tensorrt-backend"))
            .cloned(),
        headers: safe,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;

    #[test]
    fn backend_metadata_supports_metal_cuda_and_legacy_tensorrt() {
        for (header, value) in [
            ("x-tts-backend", "MLX-Metal"),
            ("x-tts-backend", "hybrid-cuda"),
            ("x-tensorrt-backend", "TensorRT-11"),
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(header, HeaderValue::from_static(value));
            headers.insert("x-tts-sample-rate", HeaderValue::from_static("48000"));
            assert_eq!(metadata(&headers).unwrap().backend.as_deref(), Some(value));
        }
    }
}
